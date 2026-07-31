# embedx

**One embedding model served across GPUs of different speed and memory — with
no configured load ratio.** That is the one reason embedx exists: the mixed
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

```bash
uv venv && uv pip install "embedx[gpu]"      # needs an NVIDIA driver; see DEPLOY.md

export EMBEDX_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
export EMBEDX_POOLING=mean                    # required, never inferred
embedx serve
```

Call it, OpenAI style:

```bash
curl http://127.0.0.1:8477/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "all-MiniLM-L6-v2", "input": ["hello world", "a longer piece of text"]}'
```

Or with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8477/v1", api_key="unused-without-EMBEDX_API_KEY")
vectors = client.embeddings.create(model="all-MiniLM-L6-v2", input=["hello world"])
```

There is also a TEI-style `POST /embed` (bare vectors), `GET /health`, and
`GET /info` (resolved config, device table, truncation counters).

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
preserved for all 256 probe rows, minimum self-cosine 0.999962, and a
maximum absolute difference from the single-GPU result of 1.95e-03 — which
is bfloat16's own resolution, not scheduler drift (`correctness_check` in
`results.json`).

| configuration | makespan | throughput | dev0 idle | dev1 idle |
|---|---|---|---|---|
| fast GPU alone | 1.085 s | 407,815 tok/s | 0.001 s | no work |
| static 50/50 by tokens | 1.614 s | 274,098 tok/s | 0.984 s | 0.000 s |
| static weighted | 1.039 s | 425,939 tok/s | 0.001 s | 0.667 s |
| **converging queue** | **0.949 s** | **466,118 tok/s** | 0.003 s | 0.007 s |

Against the fast GPU alone, the converging queue is **14.30% faster**
(1.143x). Against a static split weighted by the same device weights embedx
computes for itself, it is **9.43% faster** (1.094x). The IQRs do not
overlap (1.035–1.041 s vs 0.948–0.974 s), so the second margin is
reproducible rather than noise — but it is single digits, and worth being
plain about why.

The static weight is a coarse architectural score: multiprocessor count
times an architecture factor, computed without running anything. It rates
the A400 at 8.1% of the pair, and on this workload that is simply too low —
given a queue to pull from, the A400 absorbed **21.5%** of the tokens
(95,242 of 442,423, 32% of the items). The weighted static split therefore
starves it and then waits: it sits idle 0.667 s of a 1.039 s run, **64% of
the run**, while the fast card finishes the backlog it should never have
been given. The converging queue configures no ratio at all and finds the
split empirically, ending with both devices idle for under 8 ms. Getting a
static split *right* requires knowing that 21.5% number in advance, per
workload, per model; the queue's claim is that you should not have to.

Guessing the split wrong is expensive in the other direction too: an even
token split is **1.49x slower than not using the second GPU at all**
(1.614 s vs 1.085 s), because the slow card is handed half the work and the
fast one waits 0.98 s for it.

Measured host-to-device bandwidth is **6.36 GiB/s** on device 0 (PCIe 3.0
x8) and **2.81 GiB/s** on device 1 (PCIe 3.0 x4) — 87% and 77% of their
respective link ceilings, with the PCIe link generation sampled across the
timed region to confirm it had trained to the host maximum first (both links
park at Gen1 when idle, so a cold measurement would understate them). Note
what those numbers do to the weights: the cards differ by **2.26x** in
transfer bandwidth while their configured weight ratio is **11.33x**. No
single static number describes this pair, which is the argument for
`EMBEDX_DEVICE_WEIGHTS` being an override you may never need rather than a
knob you must tune — the queue reaches the right split whether the weights
are calibrated or not.

## Configuration

All settings come from, in order of precedence: CLI flags > `EMBEDX_*`
environment variables > a TOML file pointed to by `EMBEDX_CONFIG` > defaults.
This table is generated from `Settings` and checked by a test, so it cannot
drift:

| Setting | Type | Default | Meaning |
| --- | --- | --- | --- |
| `EMBEDX_MODEL_ID` | str | **required** | Hugging Face model id or local path. |
| `EMBEDX_REVISION` | str or None | None | Model revision: branch, tag, or commit. |
| `EMBEDX_POOLING` | cls / mean / last_token | **required** | Pooling strategy. Required on purpose: a wrong pooling produces plausible garbage vectors, so it is never inferred. |
| `EMBEDX_NORMALIZE` | bool | True | L2-normalize embeddings after pooling. |
| `EMBEDX_DTYPE` | auto / float32 / float16 / bfloat16 | 'auto' | Compute dtype; auto resolves by device capability (bf16/fp16/fp32). |
| `EMBEDX_MAX_SEQ_LEN` | int or None | None | Max sequence length in tokens; longer inputs are truncated and counted. |
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

`embedx check` preflights the configuration and device availability (exit
code for systemd); `embedx info` prints the resolved config and ranked device
table without loading a model.

## Deployment

Systemd unit (shipped as package data), env-file layout, Tailscale/UFW
binding, API keys, running alongside Ollama, troubleshooting: see
[DEPLOY.md](DEPLOY.md).

## Limitations

Stated up front rather than discovered:

- **CUDA only.** No CPU serving; `embedx serve` and `embedx check` fail
  plainly on a GPU-less host.
- **No pre-tokenized input.** `/v1/embeddings` rejects token arrays
  (`list[int]`) with an explicit error; send text.
- **`usage` counts tokens with the model's own tokenizer** (clamped at
  `max_seq_len`), which will not match OpenAI's tokenizers.
- **One model per server instance.** Run one embedx per model, on different
  ports.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 Gabin Taibi.
