# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-02

**This is a breaking release. Existing configuration will not start.**
`Settings` no longer has `model_id`, `pooling`, `dtype` or `max_seq_len`,
and the matching `EMBEDX_MODEL_ID`, `EMBEDX_POOLING`, `EMBEDX_DTYPE` and
`EMBEDX_MAX_SEQ_LEN` variables are now startup errors rather than ignored
values. A model is named per request instead, in the required `model`
field, and its pooling is resolved once, on that model's first load. See
[Migration](#migration) below.

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
- Bounded admission. `EMBEDX_MAX_CONCURRENT_REQUESTS` (default 8) caps
  in-flight embedding requests; a request queued longer than
  `EMBEDX_REQUEST_QUEUE_TIMEOUT_S` (default 30s) returns 503 instead of
  queueing indefinitely. Previously nothing capped in-flight requests at
  all.
- `EMBEDX_MAX_CONCURRENT_LOADS` (default 2) separately caps simultaneous
  cold model loads, which contend for VRAM and PCIe bandwidth. Deliberately
  a second semaphore rather than a share of the request cap: merging them
  would let slow cold loads starve requests to already-resident models. A
  load cap above the request cap is rejected at startup, since it could
  never bind.
- `GET /info` reports live `in_flight_requests` and `in_flight_loads` next
  to both caps and the queue timeout, and takes no request slot itself.
- `EMBEDX_DEFAULT_KEEP_ALIVE_S` (600s) and `EMBEDX_MAX_CONCURRENT_LOADS` (2)
  keep their values, but the values are now measured rather than reasoned
  about. `dev/04_model_load_latency.ipynb` decomposes a cold load into disk,
  host-RAM, PCIe and kernel-warmup stages: for a 7.49 GiB model the cold
  load is 9.363s, of which the disk read is 68% and the PCIe copy only 14%,
  dropping to 2.981s when the page cache is still warm. 600s puts the
  worst-case reload overhead at 1.6% of an idle window against 15.6% at 60s.
  Two concurrent cold loads cost each other ~15% on one device and nothing
  measurable across two, while finishing 1.7x/1.4x sooner — which justifies
  2 and rules out 1. Neither default changed; the placeholder justifications
  in `config.py` did.
- **Docker deployment.** `Dockerfile`, `docker-compose.yml`, `.dockerignore`
  and `.env.example`: an additional deployment path, not a replacement. The
  systemd unit and its docs are unchanged. Multi-stage, `uv`-installed,
  non-root, ~12 GB.
- **The runtime stage ships `gcc` and `libc6-dev` on purpose.** torch routes
  RoPE-family models through a Triton kernel that JIT-compiles a C helper at
  *first inference*, not at load, so without a compiler the model loads
  cleanly, registers as resident, and then every request raises. Python
  headers are not needed separately — the image's standalone CPython ships
  its own — but `gcc` without `libc6-dev` fails on a missing `stdlib.h`.
- **The build fails if the torch wheel lacks `sm_120`**, rather than letting
  it surface as `no kernel image is available for execution on the device`
  at first inference. Base is CUDA 12.8: the floor for Blackwell and
  deliberately the oldest CUDA that clears it, which keeps the **minimum
  host driver at 525+** rather than the 580+ that CUDA 13.x would demand.
- `CUDA_IMAGE` and `TORCH_INDEX_URL` are build arguments. With the task-18
  `Accelerator` seam, those two are now the whole story for adding a ROCm or
  XPU target — a new build target, not a Dockerfile rewrite. None is
  provided today.
- HF and Triton caches are **bind mounts to host storage**, not named
  volumes: the measured disk read is 68% of a cold load, so overlay or
  network storage there is the one place containerization costs real time.
  Both must be owned by the container uid, which `EMBEDX_UID`/`EMBEDX_GID`
  make configurable.
- Published to `127.0.0.1` only; healthcheck on `/health`, which takes no
  request slot and touches no model state.
- Platform support is stated rather than implied: Linux + NVIDIA is first
  class, Windows + NVIDIA works via WSL2 with friction, **macOS is not
  supported at all**.
- CI does not build the image, deliberately, and the CI config now says why.

### Fixed

Both of the first two were found in production, not in the test suite.

- **Concurrent requests no longer fail with `RuntimeError: Already
  borrowed`.** HF fast tokenizers are Rust objects with interior
  mutability — every call reconfigures truncation and padding — so two
  overlapping calls on one instance fail its borrow check. The engine's
  per-backend lock did not cover it: the scheduler calls `length_fn` on
  the requesting thread, outside that lock, while another request's worker
  is inside `embed`, and every engine is wired to `backends[0].length_fn`.
  Reproduced with 8 concurrent clients against a sentence-transformers
  checkpoint, where it surfaced as a 500.
- The batching path now has a tokenizer of its own rather than sharing one
  under a lock. On the sentence-transformers path the lock has to span the
  whole of `encode()`, which tokenizes internally, so a single tokenizer
  would make each request's batching queue behind another request's GPU
  forward pass — once per uncached input. The copy costs a few MB; if it
  fails, both paths fall back to one tokenizer under one lock, which is
  slower under load and never wrong.
- **A model is embedded once before it is allowed to become resident.** If
  that fails, its backends are torn down, device memory is returned, and the
  load fails with `SmokeTestFailedError` (503) with the real cause chained —
  rather than leaving a model that `/info` reports as healthy and that
  raises on every request. torch JIT-compiles Triton kernels at *first
  inference*, not at load, so a host missing a C toolchain produced exactly
  that shape of failure; it is the worst state in the system, because
  nothing looks wrong.
- The same check rejects a backend returning a malformed or zero-width
  array, which the Engine alone would pass through to the caller as empty
  embeddings.
- `EMBEDX_SMOKE_TEST_ON_LOAD` (default on) turns it off. This is **not free
  latency**: the first-inference kernel warmup is paid either way — task 17
  measured it at 2.29s for Qwen3-4B — so this reorders the cost onto the
  load, where it is expected and attributable, instead of onto whichever
  user sends the first request.
- The CI packaging job smoke-tested `embedx info --model-id ... --pooling
  ...`, flags removed above, and so had failed on every commit since that
  removal. It is the end-to-end proof that the core package installs and
  imports without torch, so that guarantee went unverified for the span. Its
  logs confirm the wheel built, installed and imported cleanly on every one
  of those runs; only the CLI invocation was stale.

### Changed

- Every CUDA-specific call now goes through an `Accelerator` Protocol in
  `embedx/gpu/vendor.py` — device enumeration and properties, the `cuda:N`
  device string, the OOM types caught during placement, `empty_cache`, the
  bf16/fp16 support rules and the `ARCH_FACTOR` score table. No behaviour
  changed: same scores, same placement, same dtype resolution, same error
  messages, and the GPU suite passes unmodified. Adding ROCm or XPU later
  is a new class implementing one Protocol rather than an edit across five
  files. An ast guard fails the build on any `torch.cuda` access, or any
  `"cuda:N"` device string, outside that one module.

### Known limitations

- Multi-GPU balancing only applies to models that fit on *every* device: a
  whole copy is placed per device and the inputs are sharded. On the
  reference pair the smaller card is 3.68 GiB, and a 7.49 GiB model loads on
  the larger card alone — correct behaviour, but the second card then
  contributes nothing to it. Size against your smallest card, not the sum.

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
