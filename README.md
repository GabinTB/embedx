# embedx

A text embedding server that runs one model across GPUs of **different speed
and memory**, keeps all of them busy, and needs no configured load ratio.

That is the specific gap it fills. [Text Embeddings
Inference](https://github.com/huggingface/text-embeddings-inference) serves one
model per GPU and leaves cross-device sharding to you — excellent when your
cards are interchangeable or you are happy running N independent servers.
[vLLM](https://github.com/vllm-project/vllm)'s tensor parallelism splits a
model across devices assuming they are homogeneous, which is the right
assumption for the datacenter fleets it targets and the wrong one for a
workstation that grew a second card. Both are good tools. Neither is aimed at
the mismatched box: a 16 GB Blackwell next to a 4 GB A400, where any fixed
split is wrong and the right one changes with the workload.

embedx replicates the model on every device, shards the *inputs* at runtime,
and lets the fast card come back for more until the work runs out.

[![CI](https://github.com/GabinTB/embedx/actions/workflows/ci.yml/badge.svg)](https://github.com/GabinTB/embedx/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

---

## Install

**There are two artifacts, and they are not two ways of getting the same
thing.** Pick by what you need:

| | You get | Install |
|---|---|---|
| **Docker** (or source with a flag) | The **server**: HTTP API, `embedx serve`, systemd unit | below |
| **PyPI** | The **library**: batching, scheduler, `Engine`, registry — importable, no HTTP | `pip install embedx-inference` |

The PyPI wheel deliberately contains **no `embedx.api`, no systemd unit, and
no `serve` subcommand**. If you `pip install embedx-inference` and type
`embedx serve`, it will not be there — `embedx --help` will not list it, and
will tell you where it went. That is not a packaging accident: the server's
value is the pinned combination of Python, torch, CUDA and a C toolchain, and
that combination is what the image ships and tests together.

### The server (Docker — the verified path)

Needs an NVIDIA driver **525+** and the [NVIDIA Container
Toolkit](DEPLOY.md#prerequisites).

```bash
git clone https://github.com/GabinTB/embedx && cd embedx
cp .env.example .env          # set EMBEDX_UID/GID to `id -u` / `id -g`
mkdir -p cache/hf cache/triton
docker compose up -d --build  # ~12 GB, ~5 min; almost all of it torch + CUDA
```

No model is baked in and none is loaded at startup. The first request naming
a model loads it:

```bash
curl http://127.0.0.1:8477/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "sentence-transformers/all-MiniLM-L6-v2",
       "pooling": "mean",
       "input": ["hello world", "a longer piece of text"]}'
```

The same call from the OpenAI Python client. **`pooling` has no field in the
OpenAI schema, so it goes in `extra_body`** — this is the single most common
thing to get stuck on:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8477/v1", api_key="unused-without-EMBEDX_API_KEY")

response = client.embeddings.create(
    model="sentence-transformers/all-MiniLM-L6-v2",
    input=["hello world"],
    extra_body={"pooling": "mean"},  # required on this model's FIRST load only
)
```

Once that model is resident, later requests may omit `pooling` entirely.
Sending a *different* one returns **409** rather than silently reloading or
ignoring you.

To run the server without Docker, install from source with the build flag —
see [DEPLOY.md](DEPLOY.md#install):

```bash
EMBEDX_BUILD_SERVER=1 uv pip install \
  "embedx-inference[gpu,server] @ git+https://github.com/GabinTB/embedx"
```

### The library

For a batch job on a multi-GPU box, where an HTTP round-trip per batch is
pure overhead:

```bash
pip install "embedx-inference[gpu]"
```

```python
from embedx.config import Pooling, Settings
from embedx.gpu.discovery import discover_devices, rank_devices
from embedx.registry import ModelRegistry

settings = Settings()
devices = rank_devices(discover_devices(settings.devices), settings.device_weights)
registry = ModelRegistry(devices, settings=settings)
registry.start()
try:
    with registry.acquire("sentence-transformers/all-MiniLM-L6-v2", pooling=Pooling.MEAN) as engine:
        vectors = engine.embed(["hello world", "a longer piece of text"])
        print(vectors.shape)  # (2, 384), sharded across every device
finally:
    registry.stop()
    registry.evict_all()
```

You get on-demand loading, TTL eviction and the same converging scheduler the
server uses — just without the server.

**Why the pip name differs from the import name:** you install
`embedx-inference` and you `import embedx`. `embedx` on PyPI has belonged to
an unrelated 2016 project since long before this one existed, and PyPI
normalises the name, so the short one was never available. The same split as
`scikit-learn`/`sklearn` and `beautifulsoup4`/`bs4`.

---

## How it works

Three mechanisms, and they compose:

**1. Length-sorted batching.** A batch costs `count × max_len`, because the
backend pads every input to the longest member. Sorting by token length
before batching makes each batch length-homogeneous, so the padding collapses.

**2. Token-budget batches.** Batches fill to a per-device *padded-token*
budget scaled to that card's memory, not to a fixed item count. A 16 GB card
gets 16,384 tokens per batch; the 4 GB card beside it gets 3,891. Item counts
cannot express that; token budgets can.

**3. A converging work-stealing queue.** One length-sorted queue, consumed
from **both ends inward**:

```
   sorted by token length ->
   [ short short short ......................... long long long ]
     ^                                                         ^
     slow device pulls from here          fast device pulls from here
                     ...frontiers meet wherever the hardware decides
```

The fast device takes the expensive long end, the slow device sweeps cheap
short items, and each simply comes back for more when it finishes. Nothing is
partitioned up front, so nothing has to be predicted. The split is an
*outcome*, not a setting.

---

## Measured results

Every figure below is transcribed from
[`dev/output/results.json`](dev/output/results.json) and
[`dev/output/model_load_results.json`](dev/output/model_load_results.json),
which the notebooks in [`dev/`](dev/) write. Nothing here was retyped out of a
cell, and a number that is not in those files was not measured.

**Hardware:** NVIDIA RTX PRO 2000 Blackwell (16 GB, cc 12.0) + NVIDIA RTX A400
(4 GB, PCIe 3.0 x4, cc 8.6), driver 610.43.02, torch 2.13.0+cu130.

### Two GPUs of different speed

From [`dev/03_real_gpu_throughput.ipynb`](dev/03_real_gpu_throughput.ipynb).
8,000 real `fancyzhx/ag_news` texts, 442,423 MiniLM tokens, mean pooling,
bfloat16, `max_batch_tokens=16384`. Medians of 5 timed runs after 2 discarded
warmups, with correctness asserted *before* any timing: 256/256 probe rows in
order, minimum self-cosine 0.999974, max absolute difference from the
single-GPU result 1.95e-03 — which is bfloat16's own resolution, not
scheduler drift.

| configuration | makespan | throughput | dev0 idle | dev1 idle |
|---|---|---|---|---|
| fast GPU alone | 1.067 s | 414,799 tok/s | 0.001 s | no work |
| static 50/50 by tokens | 1.577 s | 280,604 tok/s | 0.958 s | 0.000 s |
| static weighted | 1.012 s | 437,234 tok/s | 0.001 s | 0.650 s |
| **converging queue** | **0.935 s** | **473,252 tok/s** | 0.004 s | 0.003 s |

**The speed margins are modest and worth stating plainly: 1.14x over the fast
GPU alone, and 1.08x over a static split weighted by embedx's own device
weights.** The IQRs do not overlap (1.006–1.019 s against 0.930–0.938 s), so
the second margin is real rather than noise, but it is single digits. Adding a
4 GB card to a 16 GB card was never going to double anything.

**The margin is not the argument. The absence of a tuning knob is.** The
static split only reached 1.012 s because it was handed a ratio. That ratio is
a coarse architectural score — multiprocessor count times an architecture
factor, computed without running anything — and it rates the A400 at 8.1% of
the pair. Given a queue to pull from instead, the A400 took **22.4%** of the
tokens (99,106 of 442,423, and a third of the items). So the weighted split
starves it and then waits on the other card: the A400 sits idle **0.650 s of a
1.012 s run, 64% of it**, while the Blackwell finishes a backlog it should
never have been given. The converging queue configures nothing and ends with
**both devices idle under 4 ms**.

Getting a static split right means knowing that 22.4% in advance — and
re-deriving it every time the model, the corpus length distribution, or a
co-tenant on the same cards changes. The queue's claim is that you should not
have to know it at all.

Guessing wrong costs more than getting it right saves: an even token split is
**1.48x slower than not using the second GPU at all** (1.577 s against
1.067 s), because the slow card gets half the work and the fast one waits
0.96 s for it.

Measured host-to-device bandwidth is **6.36 GiB/s** on device 0 and **2.83
GiB/s** on device 1 — 87% and 77% of their respective PCIe ceilings, with the
link generation sampled across the timed region to confirm it had trained up
first. Note what that does to the weights: the cards differ by **2.25x** in
transfer bandwidth while their configured weight ratio is **11.33x**. No
single static number describes this pair, which is why `EMBEDX_DEVICE_WEIGHTS`
is an override you will probably never need rather than a knob you must tune.

### Padding waste

From [`dev/02_padding_savings.ipynb`](dev/02_padding_savings.ipynb),
`fancyzhx/ag_news` train[:2000] — 112,980 real tokens, median 53, p95 102.
Waste is the fraction of compute spent embedding padding.

| strategy | waste |
|---|---|
| fixed batch of 8, arrival order | 38.0% |
| fixed batch of 32, arrival order | 54.8% |
| fixed batch of 128, arrival order | 67.5% |
| length-sorted, 1,024-token budget | 1.0% |
| length-sorted, 16,384-token budget | 13.3% |
| length-sorted, 65,536-token budget | 35.9% |

Note that waste *grows* with the budget — a wider budget spans a wider length
range per batch — so the budget is chosen for GPU utilisation, not for
padding. The degenerate case makes the point in one line: 31 texts of length
10 plus one of length 2,000, in a fixed batch of 32, cost 64,000 padded tokens
against 2,310 real ones — **96.4% waste**. Length-sorted batching puts the
long document in a batch of its own and the waste goes to **0%**.

### What a cold model load actually costs

From [`dev/04_model_load_latency.ipynb`](dev/04_model_load_latency.ipynb).
Medians, each run in a fresh process with the checkpoint's page cache
verifiably evicted. For Qwen3-Embedding-4B (7.49 GiB) on the Blackwell card:

| stage | time | share |
|---|---|---|
| disk → OS page cache | **6.382 s** | 68% |
| host RAM → VRAM (PCIe) | 1.316 s | 14% |
| first-inference kernel warmup | 1.193 s | 13% |
| page cache → host RAM | 0.473 s | 5% |
| **total** | **9.363 s** | |

It is not the PCIe transfer people expect. Reading the checkpoint off disk
dominates, and that is exactly the part a warm page cache removes: reloading
the same model while its files are still cached costs **2.981 s**. The PCIe
copy really is bus-bound (5.69 GiB/s against a 6.36 GiB/s measured ceiling),
so its 14% is irreducible — but it is only 14%. Small models invert the
picture entirely: MiniLM loads in **0.548 s**, of which 46% is kernel autotune
and 2% is PCIe.

This is why `EMBEDX_DEFAULT_KEEP_ALIVE_S` defaults to 600 s: at a 60 s TTL an
idle lull would spend 15.6% of itself reloading; at 600 s it is 1.6%.

---

## Model support

Any **sentence-transformers** checkpoint or plain **`AutoModel`** encoder,
named per request:

- from the Hugging Face hub (`sentence-transformers/all-MiniLM-L6-v2`,
  `intfloat/e5-small-v2`, `Qwen/Qwen3-Embedding-4B`, …);
- from a **local path**, which makes private checkpoints and your own
  fine-tunes first-class rather than an afterthought — mount the directory and
  name it in the request;
- pinned to a branch, tag or commit with `EMBEDX_REVISION`.

Different requests may name different models; each is loaded on first use and
evicted when idle.

**Pooling is required on a model's first load and is never inferred.** This is
deliberate and it is the one piece of friction embedx will not remove for you:
the wrong pooling does not raise, does not warn, and does not look wrong. It
returns correctly-shaped, plausibly-distributed vectors that are silently
useless, and you find out weeks later from bad retrieval. So it is stated
once, explicitly, and recorded in `/info` for every resident model.

**Weights must be safetensors.** A server that loads whatever a request names
cannot also run pickle deserialization, which executes arbitrary code at load
time. A checkpoint with no safetensors is refused, and so is one whose format
cannot be checked.

---

## API

| endpoint | auth | purpose |
|---|---|---|
| `POST /v1/embeddings` | yes | OpenAI-compatible. `encoding_format: float \| base64`. |
| `POST /embed` | yes | TEI-style, returns bare vectors. Unlike TEI, `model` is required. |
| `GET /health` | no | Liveness. Touches no model state, takes no request slot. |
| `GET /info` | yes | Server config, live concurrency counters, and every resident model with its pooling, dtype, devices, idle time and truncation count. |

Both embedding endpoints accept four fields beyond the OpenAI schema. **All
four apply only to a model's first load** — once it is resident they are
ignored, because the model is shared by every caller and one request does not
get to reconfigure it underneath the others:

| field | meaning |
|---|---|
| `pooling` | `cls` / `mean` / `last_token`. Required on first load. |
| `dtype` | `auto` / `float32` / `float16` / `bfloat16`. |
| `max_seq_len` | Truncation length in tokens. |
| `keep_alive` | Seconds resident after last use. `0` unloads as soon as the request finishes. |

Status codes worth knowing:

| code | when |
|---|---|
| **400** | First load with no `pooling`; pickle-only weights; `dimensions` not matching the model's output width. |
| **401** | `EMBEDX_API_KEY` is set and the bearer token is missing or wrong. |
| **409** | `pooling` conflicts with what this model is already loaded with. Not reloaded, not ignored. |
| **413** | Body over `EMBEDX_MAX_REQUEST_BYTES`. |
| **503** | The model fit on no device; it loaded but failed its smoke test; the residency cap is reached with every model busy; the weight format could not be verified; or the request queued past `EMBEDX_REQUEST_QUEUE_TIMEOUT_S`. All retryable — the request was fine, the server currently cannot. |

Every error uses the OpenAI error envelope. Tracebacks are logged
server-side and never appear in a response body.

---

## Configuration

Precedence: CLI flags > `EMBEDX_*` environment variables > a TOML file at
`$EMBEDX_CONFIG` > defaults. This table is rendered from `Settings` and
asserted verbatim by a test, so it cannot drift from the code:

| Setting | Type | Default | Meaning |
| --- | --- | --- | --- |
| `EMBEDX_REVISION` | str or None | None | Model revision: branch, tag, or commit. |
| `EMBEDX_NORMALIZE` | bool | True | L2-normalize embeddings after pooling. |
| `EMBEDX_WRAPPING` | str or None | None | Template wrapping every input, e.g. "Q: {text}"; must contain '{text}' exactly once. Unset means inputs are passed through untouched. |
| `EMBEDX_HOST` | str | '127.0.0.1' | Bind address; loopback by default. |
| `EMBEDX_PORT` | int | 8477 | Bind port (1024-65535). |
| `EMBEDX_API_KEY` | str or None | None | Bearer API key; unset disables auth entirely. |
| `EMBEDX_MAX_REQUEST_ITEMS` | int | 2048 | Maximum number of inputs per request. |
| `EMBEDX_MAX_REQUEST_BYTES` | int | 32000000 | Maximum request body size in bytes. |
| `EMBEDX_LOG_LEVEL` | str | 'INFO' | Log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL. |
| `EMBEDX_DEVICES` | list[int] or None | None | CUDA device indices (list or "0,1"); default: all visible. |
| `EMBEDX_MAX_BATCH_TOKENS` | int | 16384 | Default per-batch padded-token budget. |
| `EMBEDX_MAX_BATCH_ITEMS` | int or None | None | Per-batch item-count cap (bounds zero-cost batches). |
| `EMBEDX_DEVICE_WEIGHTS` | dict[int, float] | {} | Per-device speed-weight overrides, e.g. "0=1.0,1=0.35". |
| `EMBEDX_DEVICE_BATCH_TOKENS` | dict[int, int] | {} | Per-device token-budget overrides, e.g. "0=16384,1=4096". |
| `EMBEDX_DEFAULT_KEEP_ALIVE_S` | float | 600.0 | Seconds an idle model stays resident before it is unloaded. |
| `EMBEDX_MAX_LOADED_MODELS` | int or None | None | Max models resident at once; unset means no cap. At the cap, a new load evicts the least-recently-used model that no request is using, rather than refusing; if every resident model is in use, the load fails instead. |
| `EMBEDX_SMOKE_TEST_ON_LOAD` | bool | True | Embed one short string before marking a model resident, so a model that loads but cannot infer fails the load instead of every later request. Costs the first-inference kernel warmup, which is paid either way. |
| `EMBEDX_MAX_CONCURRENT_REQUESTS` | int | 8 | Max embedding requests in flight at once; the rest queue. |
| `EMBEDX_REQUEST_QUEUE_TIMEOUT_S` | float | 30.0 | Seconds a queued request waits for a slot before returning 503. |
| `EMBEDX_MAX_CONCURRENT_LOADS` | int | 2 | Max cold model loads running at once; warm requests never queue here. |

There is no `EMBEDX_MODEL_ID` and no `EMBEDX_POOLING`: a server is configured
without naming a checkpoint, and every request carries its own `model`. Both
variables are rejected at startup rather than ignored, so an upgraded box
cannot appear to honour configuration it is silently dropping.

`embedx info` prints the resolved configuration and the ranked device table
without loading anything. `embedx check` is the systemd preflight, and
`embedx check --warm <model> --warm-pooling <pooling>` additionally loads that
model once and unloads it, leaving nothing resident.

---

## Deployment

Two supported paths, both single-host: **Docker** when you want the dependency
stack solved for you, **systemd** when the host already has the right Python
and toolchain and you would rather not run a 12 GB image.

[**DEPLOY.md**](DEPLOY.md) covers both end to end: prerequisites, the cache
volumes (which are the part that actually affects performance), the systemd
unit, API keys and network binding, GPU selection, running alongside Ollama,
and a troubleshooting section for the failures this project actually hit.

---

## Limitations

Read this section before the feature list. It will tell you faster whether
embedx fits.

- **CUDA only.** There is a vendor seam — every CUDA-specific call sits behind
  one `Accelerator` protocol, enforced by a test — but **no ROCm or XPU
  implementation exists**. The seam makes one a new class rather than a
  rewrite; it does not make one exist. Apple MPS is explicitly out of scope:
  unified memory means there is no discrete VRAM budget to scale token budgets
  against.
- **No CPU serving.** `embedx serve` and `embedx check` fail plainly on a
  GPU-less host.
- **Linux + NVIDIA is the only first-class platform.** Windows works through
  WSL2 with friction. **macOS is not supported at all** — Apple dropped NVIDIA
  drivers after 10.13 and Docker Desktop on macOS has no GPU passthrough.
- **Multi-GPU balancing only helps models that fit on _every_ device.** embedx
  replicates the model per device and shards the inputs, so the smallest card
  sets the ceiling. On a 16 GB + 4 GB pair, anything above roughly **3.5 GB at
  bf16** does not fit the A400: the load OOMs there, placement keeps whichever
  devices succeeded, and the model is served from the Blackwell alone.
  Measured, not estimated — Qwen3-Embedding-4B (7.49 GiB) does exactly this.
  Nothing breaks, but **the scheduler then has nothing to balance and the
  second card contributes nothing**. If you are sizing a mismatched pair, your
  smallest card's capacity is the number that matters, not the total.
- **One model per request.** There is no batching across different models.
- **A cold load blocks the request that triggered it.** No asynchronous pull
  endpoint; naming an unloaded model means waiting out the load above.
- **No pre-tokenized input.** OpenAI accepts token arrays; embedx rejects them
  with an explicit error. Send text.
- **`usage` token counts come from the model's own tokenizer** (clamped at
  `max_seq_len`), so they will not match OpenAI's counts.
- **RoPE-family models need a C toolchain at runtime.** torch routes them
  (Qwen3, for example) through a Triton kernel that JIT-compiles a C helper at
  *first inference*, not at load — so on a host without `gcc` and libc headers
  the model loads cleanly, reports resident, and then fails every request.
  This is what the Docker image solves. On bare metal it is on you; the
  load-time smoke test at least turns it into a failed load rather than a
  permanently broken resident model.
- **Throughput was measured on exactly one machine**, on one corpus, with one
  model. The two cards differ by roughly 11:1, which caps what a second device
  can add. No claim is made about other pairings, larger models, or more than
  two GPUs.

---

## Contributing

The gate, which must be green on every commit:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/embedx
uv run pytest -m "not gpu"
```

`make gate` runs all four. Set up with `uv sync --extra dev`; add `--extra
gpu` on a CUDA host to run the GPU-marked tests.

Two conventions worth knowing before you send anything: **one task, one
commit** — changes are not batched and tests are not deferred to a follow-up;
and **core stays importable without torch**, with the heavy backends behind
the `gpu` extra and imported lazily. Both are enforced by tests rather than by
review. This is a solo project, so there is no PR ceremony — open an issue
describing the problem before writing code for anything non-trivial, so we do
not both build it.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 Gabin Taibi.
