# embedx

**Many embedding models, loaded on demand, served across GPUs of different
speed and memory — with no configured load ratio.** That is the one reason embedx exists: the mixed
box. [TEI](https://github.com/huggingface/text-embeddings-inference) runs one
model per GPU and leaves the request sharding to you; vLLM's tensor
parallelism assumes homogeneous devices. embedx assumes the opposite and
balances at runtime.

## How it balances

- **Length-sorted batching** — inputs are sorted by token length so each batch
  is length-homogeneous and padding (cost `count × max_len`) is minimal.
- **Token-budget batches** — batches fill to a per-device *padded-token*
  budget, scaled to each card's memory, instead of a fixed item count.
- **Converging work-stealing queue** — one sorted queue consumed from both
  ends inward: fast devices claim from the expensive long end, slow devices
  sweep the short end, and the frontiers meet wherever the hardware dictates.
  A faster card simply comes back for more; nothing is partitioned up front.

## Quickstart

**Docker** (recommended — the Python / CUDA-wheel / Triton dependency problem
is solved inside the image). Needs the [NVIDIA Container
Toolkit](DEPLOY.md#prerequisites) and driver **525+**:

```bash
cp .env.example .env          # set EMBEDX_UID/GID to `id -u` / `id -g`
mkdir -p cache/hf cache/triton
docker compose up -d --build  # ~12 GB, ~5 min; almost all of it torch + CUDA
```

**Or on the host**, if it already has a matching Python, a C compiler and the
driver. The server installs from source, because that is where it ships:

```bash
uv venv
EMBEDX_BUILD_SERVER=1 uv pip install \
  "embedx-inference[gpu,server] @ git+https://github.com/GabinTB/embedx"

embedx serve                                 # starts with no model loaded
```

`EMBEDX_BUILD_SERVER=1` is what includes the HTTP layer in the build; without
it you get the library and `embedx serve` will not exist. `[server]` supplies
its dependencies, the flag supplies its modules.

**Or as a library**, for a batch job that wants the multi-GPU scheduling
without an HTTP round-trip in the middle:

```bash
uv pip install "embedx-inference[gpu]"       # then: import embedx as ebx
```

The distribution is `embedx-inference`; the import name is plain `embedx`.
They differ because `embedx` on PyPI has belonged to an unrelated 2016
project since before this one existed — the same split as
`scikit-learn`/`sklearn`. The PyPI wheel is the **library** (batching,
scheduler, `Engine`, device ranking, model registry, config); the HTTP server
ships with the Docker image or a source install, so `embedx serve` exists on
those and not on a plain `pip install`.

Name the model in the request. The first one to name it loads it, and
`pooling` is required on that first load — it is never inferred, because the
wrong choice returns plausible vectors that are silently wrong:

```bash
curl http://127.0.0.1:8477/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "sentence-transformers/all-MiniLM-L6-v2",
       "pooling": "mean",
       "input": ["hello world", "a longer piece of text"]}'
```

Every later request for that model can omit `pooling`; sending a *different*
one returns 409 rather than quietly reloading or ignoring you. A model is
unloaded once it has been idle for `EMBEDX_DEFAULT_KEEP_ALIVE_S`, and
`keep_alive` in the request overrides that per model.

With the OpenAI Python client (which has no field for `pooling`, so load the
model once with curl, or use `extra_body`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8477/v1", api_key="unused-without-EMBEDX_API_KEY")
vectors = client.embeddings.create(
    model="sentence-transformers/all-MiniLM-L6-v2",
    input=["hello world"],
    extra_body={"pooling": "mean"},
)
```

There is also a TEI-style `POST /embed` (bare vectors; `model` required,
unlike TEI), `GET /health`, and `GET /info` (server config, every resident
model with its pooling, devices, idle time and truncation count, and live
concurrency counts).

In-flight requests and cold model loads are separately capped
(`EMBEDX_MAX_CONCURRENT_REQUESTS`, `EMBEDX_MAX_CONCURRENT_LOADS`) so a
request to a warm model never queues behind a cold load of another one; past
`EMBEDX_REQUEST_QUEUE_TIMEOUT_S` a queued request gets a 503 rather than
waiting forever.

## Measured results

### Padding waste

(From [`dev/02_padding_savings.ipynb`](dev/02_padding_savings.ipynb),
`fancyzhx/ag_news` train[:2000] tokenized with MiniLM — 112,980 real tokens,
median 53, p95 102 — waste = padded − real tokens as a fraction of padded):
fixed item-count batching in arrival order wastes **38.0%** of compute at
batch size 8, **54.8%** at 32, **67.5%** at 128. Length-sorted token-budget
batching wastes **1.0%** at a 1,024-token budget, **13.3%** at 16,384,
**35.9%** at 65,536. The degenerate case is the argument in one line: one
2,000-char document in a fixed batch of 32 short texts wastes **96.4%** of
the batch; length-sorted batching isolates it and wastes **0%**.

### Scheduling balance

(Simulation, [`dev/01_scheduling_visualization.ipynb`](dev/01_scheduling_visualization.ipynb),
real `Scheduler` + the length-sensitive fake cost model.) With a 3× speed
gap, the fast worker takes **78.6%** of tokens and both workers finish within
**0.8%** of each other; across 1×–10× speed ratios the fast worker's item
share adapts from 0.21 to 0.72 with zero configuration.

### Two-GPU throughput on real hardware

From [`dev/03_real_gpu_throughput.ipynb`](dev/03_real_gpu_throughput.ipynb).
Every figure below is transcribed from
[`dev/output/results.json`](dev/output/results.json), which that notebook
writes; nothing here was retyped out of a cell.

**Hardware:** NVIDIA RTX PRO 2000 Blackwell (16 GB, cc 12.0) + NVIDIA RTX
A400 (4 GB, PCIe 3.0 x4, cc 8.6), driver 610.43.02, torch 2.13.0+cu130
(CUDA 13.0).

**Corpus:** `fancyzhx/ag_news` train[:8000] — **8,000 real news headlines**,
442,423 MiniLM tokens, median 53 per text. Real text, deliberately: an
earlier draft of the notebook fell back to generated filler when the dataset
id stopped resolving, and every number that run produced has been discarded
rather than reused. MiniLM-L6-v2, mean pooling, `dtype=auto` (bfloat16 on
both cards), `max_batch_tokens=16384`. Medians of 5 timed runs after 2
discarded warmups, with correctness asserted *before* any timing: order
preserved for all 256 probe rows, minimum self-cosine 0.999974, and a
maximum absolute difference from the single-GPU result of 1.95e-03 — which
is bfloat16's own resolution, not scheduler drift (`correctness_check` in
`results.json`).

| configuration | makespan | throughput | dev0 idle | dev1 idle |
|---|---|---|---|---|
| fast GPU alone | 1.067 s | 414,799 tok/s | 0.001 s | no work |
| static 50/50 by tokens | 1.577 s | 280,604 tok/s | 0.958 s | 0.001 s |
| static weighted | 1.012 s | 437,234 tok/s | 0.001 s | 0.650 s |
| **converging queue** | **0.935 s** | **473,252 tok/s** | 0.004 s | 0.003 s |

Against the fast GPU alone, the converging queue is **14.09% faster**
(1.141x). Against a static split weighted by the same device weights embedx
computes for itself, it is **8.24% faster** (1.082x). The IQRs do not
overlap (1.006–1.019 s vs 0.930–0.938 s), so the second margin is
reproducible rather than noise — but it is single digits, and worth being
plain about why.

The static weight is a coarse architectural score: multiprocessor count
times an architecture factor, computed without running anything. It rates
the A400 at 8.1% of the pair, and on this workload that is simply too low —
given a queue to pull from, the A400 absorbed **22.4%** of the tokens
(99,106 of 442,423, 33% of the items). The weighted static split therefore
starves it and then waits: it sits idle 0.650 s of a 1.012 s run, **64% of
the run**, while the fast card finishes the backlog it should never have
been given. The converging queue configures no ratio at all and finds the
split empirically, ending with both devices idle for under 4 ms. Getting a
static split *right* requires knowing that 22.4% number in advance, per
workload, per model; the queue's claim is that you should not have to.

Guessing the split wrong is expensive in the other direction too: an even
token split is **1.48x slower than not using the second GPU at all**
(1.577 s vs 1.067 s), because the slow card is handed half the work and the
fast one waits 0.96 s for it.

Measured host-to-device bandwidth is **6.36 GiB/s** on device 0 (PCIe 3.0
x8) and **2.83 GiB/s** on device 1 (PCIe 3.0 x4) — 87% and 77% of their
respective link ceilings, with the PCIe link generation sampled across the
timed region to confirm it had trained to the host maximum first (both links
park at Gen1 when idle, so a cold measurement would understate them). Note
what those numbers do to the weights: the cards differ by **2.25x** in
transfer bandwidth while their configured weight ratio is **11.33x**. No
single static number describes this pair, which is the argument for
`EMBEDX_DEVICE_WEIGHTS` being an override you may never need rather than a
knob you must tune — the queue reaches the right split whether the weights
are calibrated or not.

### Cold model loads

From [`dev/04_model_load_latency.ipynb`](dev/04_model_load_latency.ipynb),
raw values in
[`dev/output/model_load_results.json`](dev/output/model_load_results.json).
Medians, each run in a fresh process with the checkpoint's page cache
verifiably evicted, on the same two-GPU host.

A cold load is not dominated by the PCIe transfer people expect. For
Qwen3-Embedding-4B (7.49 GiB) on the Blackwell card:

| stage | time | share |
|---|---|---|
| disk → OS page cache | **6.382 s** | 68% |
| host RAM → VRAM (PCIe) | 1.316 s | 14% |
| first-inference kernel warmup | 1.193 s | 13% |
| page cache → host RAM | 0.473 s | 5% |
| **total** | **9.363 s** | |

Reading the checkpoint off disk is the largest stage by a wide margin, and it
is the one a warm page cache removes: reloading the same model while its
files are still cached costs **2.981 s** instead. The PCIe copy is genuinely
bus-bound (5.69 GiB/s against a measured 6.36 GiB/s ceiling), so that 14% is
irreducible — but it is only 14%. Small models invert the picture: MiniLM
loads in **0.548 s**, of which 46% is kernel autotune and 2% is PCIe.

`EMBEDX_DEFAULT_KEEP_ALIVE_S` defaults to 600 s against these numbers: at
60 s a lull would spend 15.6% of its time reloading, at 600 s it is 1.6%.

### Scope limit: the balancing needs the model on every card

embedx replicates a whole model per device and shards the *inputs*. That only
works for a model small enough to fit on **every** device you give it.

On this pair the smaller card is an RTX A400 with **3.68 GiB**, and
Qwen3-Embedding-4B (7.49 GiB) does not fit it — the load OOMs on that device
and the model is served from the Blackwell card alone. Nothing breaks:
device-by-device placement keeps whichever devices succeed. But for such a
model **the converging scheduler has nothing to balance, and the second card
contributes nothing.**

So the throughput results above apply to models under your smallest card's
capacity. Above that, embedx is a single-GPU server with a model registry. If
you are sizing this for a mismatched pair, that threshold — not the total
VRAM across both cards — is the number that matters.

## Configuration

All settings come from, in order of precedence: CLI flags > `EMBEDX_*`
environment variables > a TOML file pointed to by `EMBEDX_CONFIG` > defaults.
This table is generated from `Settings` and checked by a test, so it cannot
drift:

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

There is no `EMBEDX_MODEL_ID` or `EMBEDX_POOLING`: a server is configured
without naming a checkpoint, and every request carries its own `model`.

`embedx check` preflights configuration and device availability (exit code
for systemd); `embedx check --warm <model_id> --warm-pooling <pooling>` also
loads that model once and unloads it, as an end-to-end preflight that leaves
nothing resident. `embedx info` prints the resolved config and ranked device
table without loading anything.

## Deployment

Two supported paths, both single-host: **Docker** when you want the
dependency stack solved for you, **systemd** when the host already has the
right Python and toolchain and you would rather not run a 12 GB image.
Dockerfile, compose file, systemd unit (shipped as package data), env-file
layout, Tailscale/UFW binding, API keys, running alongside Ollama,
troubleshooting: see [DEPLOY.md](DEPLOY.md).

Compute overhead under the NVIDIA Container Toolkit is roughly 1–2% and not
worth worrying about. Disk I/O is the part that can actually regress, and
only if the model cache volume is misconfigured — the cold-load numbers
above are why.

### Platform support

Stated plainly, because "cross-platform" would be a lie:

| Platform | Status |
|---|---|
| **Linux + NVIDIA** | First class. What everything here is tested on. |
| **Windows + NVIDIA** | Works via WSL2, with friction. |
| **macOS** | **Not supported.** Apple dropped NVIDIA drivers after 10.13, and Docker Desktop on macOS has no GPU passthrough. |

The container base is CUDA 12.8, which is the floor for `sm_120` (Blackwell)
and implies a **minimum host driver of 525**. That is deliberately the
oldest CUDA that supports Blackwell rather than the newest available: CUDA
13.x would require driver 580+ and exclude most users for no gain.

## Limitations

Stated up front rather than discovered:

- **CUDA only.** No CPU serving; `embedx serve` and `embedx check` fail
  plainly on a GPU-less host.
- **No pre-tokenized input.** `/v1/embeddings` rejects token arrays
  (`list[int]`) with an explicit error; send text.
- **`usage` counts tokens with the model's own tokenizer** (clamped at
  `max_seq_len`), which will not match OpenAI's tokenizers.
- **Weights must be safetensors.** A server that loads whatever a request
  names cannot also run pickle deserialization; a checkpoint without
  safetensors is refused, and so is one whose format cannot be checked.
- **A cold load blocks the request that triggered it.** There is no
  asynchronous pull endpoint yet.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 Gabin Taibi.
