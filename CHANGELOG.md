# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
