# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — BREAKING

embedx serves many models, loaded on demand, instead of one chosen at
startup. A server is now configured without naming a checkpoint.

- **Removed from `Settings`:** `model_id`, `pooling`, `dtype`,
  `max_seq_len`, and therefore `EMBEDX_MODEL_ID`, `EMBEDX_POOLING`,
  `EMBEDX_DTYPE` and `EMBEDX_MAX_SEQ_LEN`. They are per-request fields now:
  `model` (required), and optional `pooling` / `dtype` / `max_seq_len`,
  which apply on a model's first load only. Setting the old variables is a
  startup error, not a no-op: pydantic-settings would otherwise ignore an
  `EMBEDX_*` variable matching no field, and a box whose env file still
  pinned `EMBEDX_MODEL_ID` would start happily while its operator believed
  it was pinned. Delete them.
- **`model` in a request is a routing key, not a label.** Naming a model
  the server has not loaded loads it, and the request blocks until that
  finishes. There is no asynchronous pull endpoint yet.
- **`pooling` is required on a model's first load** and refused with 400 if
  absent; a later request naming the same model with a different pooling
  gets 409. The rule that pooling is never inferred did not go away, it
  moved from configuration time to load time.
- **`POST /embed` now requires `model`.** TEI has no such field, but embedx
  has no configured model to fall back to.
- **`GET /info` is per-model:** server-wide config plus one entry per
  resident model (pooling, dtype, devices, idle time, in-flight references,
  truncation count). Zero loaded models is an empty list, not an error.
- **`embedx serve` starts with nothing loaded** and no longer builds an
  engine eagerly; `build_engine` is gone.
- **`embedx check` no longer preflights a model.** `--warm <model_id>`
  with `--warm-pooling` does a real load-and-unload cycle instead, leaving
  nothing resident.
- CLI options `--model-id`, `--pooling`, `--dtype` and `--max-seq-len` are
  removed from `serve`, `info` and `check`.

### Added

- On-demand model registry with per-model-id load locking, device-by-device
  placement, reference-counted eviction safety and a TTL reaper.
- `EMBEDX_DEFAULT_KEEP_ALIVE_S` (default 600s): idle seconds before a model
  is unloaded. Overridable per request with `keep_alive`; `keep_alive: 0`
  unloads as soon as the request finishes.
- `EMBEDX_MAX_LOADED_MODELS` (default unset, no cap): at the cap, a new load
  evicts the least-recently-used model that no request is using; if every
  resident model is in use the load fails with 503 rather than evicting one
  mid-request.
- Dynamically loaded weights must be safetensors. A pickle-only checkpoint
  is refused with 400; a checkpoint whose format cannot be verified at all
  is refused with 503, after falling back to the local HF cache so an
  air-gapped host still serves what it has already downloaded.
- `EMBEDX_WRAPPING`: a template applied to every input, e.g. `"Q: {text}"`.

### Migration

```diff
  # /etc/embedx/embedx.env
- EMBEDX_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
- EMBEDX_POOLING=mean
+ # nothing required; models are named per request
```

```diff
  {
-   "model": "all-MiniLM-L6-v2",
+   "model": "sentence-transformers/all-MiniLM-L6-v2",
+   "pooling": "mean",
    "input": ["hello world"]
  }
```

`pooling` is needed only on the request that first loads a given model.

## [0.1.0] - 2026-07-31

### Added

- Length-sorted token-budget batching: `pack_count`, the single
  direction-aware home of the padded-cost model (`count × max_len`), and
  `make_batches` with an optional per-batch item cap.
- Converging work-stealing scheduler: one sorted queue with a lock-guarded
  `(lo, hi)` cursor pair, consumed from both ends by heterogeneous workers;
  exactly-once claims, `None` as the only exhaustion signal, fastest-half
  workers assigned to the expensive end.
- Multi-worker engine: one thread per backend over a shared scheduler,
  per-backend locks against VRAM oversubscription under concurrent requests,
  output order preserved by construction, worker failures re-raised with the
  device index and never returned as partial results.
- GPU discovery and static ranking without benchmarking
  (`multi_processor_count × ARCH_FACTOR`), config-overridable per-device
  weights, and memory-scaled per-device token budgets.
- Hugging Face / sentence-transformers CUDA backend behind the `gpu` extra:
  lazy torch imports (core stays importable without torch, enforced by an ast
  guard test), explicit mask-aware pooling (`cls` / `mean` / `last_token`),
  configured pooling always winning over checkpoint-declared pooling with a
  warning on disagreement, dtype auto-resolution by device capability,
  truncation counting, and a token-length cache shared across devices.
- OpenAI-compatible `POST /v1/embeddings` (float and base64 encodings, real
  token usage, enforced response models), TEI-style `POST /embed`,
  `GET /health`, `GET /info` with per-device budgets and truncation counters;
  bearer auth, request size limits covering chunked bodies, one error
  envelope for every failure path.
- Configuration via pydantic-settings: `EMBEDX_*` env variables, TOML file
  via `EMBEDX_CONFIG`, precedence CLI > env > file > defaults; `pooling`
  required by design; override strings for per-device weights and budgets;
  exposure warning when bound beyond loopback without an API key.
- CLI: `embedx serve`, `embedx info` (configuration inspector), and
  `embedx check` (systemd preflight), with explicit-only CLI-to-settings
  forwarding so environment and file values survive unset flags.
- Deployment: hardened systemd unit shipped as package data (with the two
  GPU-breaking hardening options documented as deliberately absent) and
  `DEPLOY.md`.
- Validation notebooks under `dev/`: scheduler convergence simulation,
  padding-waste comparison, and the two-GPU throughput benchmark harness.
- CI: ruff, mypy, pytest gate plus a packaging job installing the built wheel
  torch-free, on Python 3.11 and 3.12.

### Known limitations

- CUDA only; no CPU serving.
- Pre-tokenized input (token arrays) is rejected.
- One model per server instance.
- Throughput is measured on exactly one machine. On an RTX PRO 2000
  Blackwell + RTX A400 host, over 8,000 real `fancyzhx/ag_news` texts, the
  converging queue is 14.30% faster than the fast GPU alone and 9.43% faster
  than a static split weighted by embedx's own device weights, with both
  devices idle under 8 ms against 0.667 s for the weighted split
  (`dev/03_real_gpu_throughput.ipynb`, raw values in
  `dev/output/results.json`). Those two cards differ by roughly 11:1, which
  caps what a second device can add; no claim is made about other pairings,
  larger models, or more than two GPUs.
