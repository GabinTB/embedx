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

**Padding waste** (from [`dev/02_padding_savings.ipynb`](dev/02_padding_savings.ipynb),
`fancyzhx/ag_news` train[:2000] tokenized with MiniLM — 112,980 real tokens,
median 53, p95 102 — waste = padded − real tokens as a fraction of padded):
fixed item-count batching in arrival order wastes **38.0%** of compute at
batch size 8, **54.8%** at 32, **67.5%** at 128. Length-sorted token-budget
batching wastes **1.0%** at a 1,024-token budget, **13.3%** at 16,384,
**35.9%** at 65,536. The degenerate case is the argument in one line: one
2,000-char document in a fixed batch of 32 short texts wastes **96.4%** of
the batch; length-sorted batching isolates it and wastes **0%**.

**Scheduling balance** (simulation, [`dev/01_scheduling_visualization.ipynb`](dev/01_scheduling_visualization.ipynb),
real `Scheduler` + the length-sensitive fake cost model): with a 3× speed
gap, the fast worker takes **78.6%** of tokens and both workers finish within
**0.8%** of each other; across 1×–10× speed ratios the fast worker's item
share adapts from 0.21 to 0.72 with zero configuration.

**Two-GPU throughput on real hardware** (from
[`dev/03_real_gpu_throughput.ipynb`](dev/03_real_gpu_throughput.ipynb), raw
values in [`dev/output/results.json`](dev/output/results.json)): RTX PRO 2000
(Blackwell, 16 GB, cc 12.0) + RTX A400 (4 GB, PCIe 3.0 x4, cc 8.6), driver
610.43.02, torch 2.13.0+cu130. 8,000 ag_news texts / 442,423 tokens through
MiniLM-L6-v2, mean pooling, bfloat16, `max_batch_tokens=16384`; medians of 5
runs after 2 discarded warmups, correctness asserted before any timing.

| configuration | makespan | throughput | idle (dev0 / dev1) |
|---|---|---|---|
| fast GPU alone | 1.085 s | 407,815 tok/s | 0.001 s / — |
| static 50/50 by tokens | 1.614 s | 274,098 tok/s | 0.984 s / 0.000 s |
| static weighted | 1.039 s | 425,939 tok/s | 0.001 s / 0.667 s |
| **converging queue** | **0.949 s** | **466,118 tok/s** | 0.003 s / 0.007 s |

The converging queue is **1.14x** the single fast GPU and **1.09x** a tuned
weighted static split. That margin is modest and the reason is structural:
these two cards differ by roughly 11:1, so a second GPU of that size cannot
buy much. What the numbers do show is balance. `device_weights` allot the
A400 8.1% of the work; allowed to pull, it took **21.5%** — which is why the
weighted split leaves it idle for 0.667 s, 64% of its own makespan, while the
converging queue idles both devices for under 8 ms. Splitting evenly is worse
than ignoring the second GPU entirely: **1.49x slower** than one card alone.
Measured H2D bandwidth differs by 2.25x (6.4 vs 2.8 GiB/s) against a
configured weight ratio of 11.3x, and the queue is indifferent to that
miscalibration — which is the tuning burden it exists to remove.

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
