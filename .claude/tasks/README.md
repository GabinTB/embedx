# embedx build plan

Commit-oriented task list. Execute in order. **One task = one commit.** Each task
below has: scope, files, tests to add, and the acceptance gate. Do not merge
tasks. Do not defer tests.

## The gate (every task)

```
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not gpu"
```

All green, plus `mypy src/embedx` once introduced in task 02. A task is done only
when the gate is green and the new tests for that task exist and pass.

## Philosophy

Build in arcs:

1. **Skeleton & tooling** (tasks 00–02): repo runs, lints, tests, CLI stub. No
   real logic yet, just green scaffolding and the testing seam.
2. **Core logic, backend-agnostic** (tasks 03–07): batching, scheduling, engine,
   config, API — all tested on CPU with the fake backend. This is where the novel
   algorithms live and get hardened. No torch required.
3. **Real backend, service, polish** (tasks 08–12): the HF/CUDA backend behind the
   `gpu` extra, GPU tests, systemd, docs, notebooks, and the convergence of
   everything into a shippable, PyPI-publishable package.
4. **Multi-model** (tasks 13–19): one server, many checkpoints, loaded on demand
   and named per request. Includes the breaking config migration that removed
   the single-model settings, the measurements that calibrate the residency and
   concurrency defaults, the vendor seam, and Docker.
5. **Publication** (tasks 20–22): what has to be true before the repo is read by
   strangers rather than by the person who wrote it.

## Task index

- `00_scaffold.md` — uv project, src layout, packaging, ruff/pytest config, CI.
- `01_testing_seam.md` — `EmbeddingBackend` protocol + fake backend + first tests.
- `02_types_and_ci_gate.md` — mypy, ruff format, CI workflow enforcing the gate.
- `03_batching.md` — length-sorted token-budget batching + full tests.
- `04_scheduling.md` — converging work-stealing queue + invariant/property tests.
- `05_config.md` — pydantic-settings config, overrides, validation + tests.
- `06_gpu_ranking.md` — device discovery + static ranking + budgets (no torch) + tests.
- `07_engine_and_api.md` — Engine wiring + FastAPI OpenAI-compatible API (fake backend) + tests.
- `08_hf_backend.md` — real transformers/sentence-transformers backend (gpu extra) + gpu tests.
- `09_cli_and_serve.md` — `embedx serve/info/check` CLI wired to real engine.
- `10_service_and_deploy.md` — systemd unit, deployment docs, security defaults.
- `11_notebooks_and_validation.md` — dev notebooks: convergence viz, padding savings, real throughput.
- `12_release.md` — version, changelog, PyPI packaging check, README polish, tag.
- `13_model_registry.md` — on-demand multi-model registry, TTL eviction, lifecycle.
- `14_api_multi_model_routing.md` — route `/v1/embeddings` and `/embed` through it.
- `15_config_breaking_migration.md` — drop `model_id`/`pooling`/`dtype`/`max_seq_len`
  from `Settings`; models are named per request (BREAKING).
- `16_concurrency_cap.md` — separate caps for in-flight requests and cold loads.
- `17_load_latency_benchmark.md` — decompose a cold load into four measured stages;
  calibrate `default_keep_alive_s` and `max_concurrent_loads` against it.
- `18_vendor_seam.md` — every CUDA-specific call behind one `Accelerator` Protocol.
- `19_docker.md` — Dockerfile, compose, the C toolchain the Triton JIT needs.
- `20_publication_audit.md` — pre-publication audit and third-party reference scrub.
- `21_pypi_library_split.md` — PyPI ships the library; the server ships via Docker.
- `22_public_readme.md` — the README a stranger evaluating embedx actually reads.

Shipped since this list was first written, and no longer "later": **base64
`encoding_format`** (landed in task 07 with the API itself — the official
OpenAI client sends base64 by default, so a float-only server would have
broken the most common caller) and **multi-model serving** (tasks 13–19, which
became the whole second half of the plan rather than a post-1.0 extra).

Still later (post-1.0, not scheduled here): reranker endpoint, Prometheus
metrics, dynamic batching across concurrent requests, an asynchronous model-pull
endpoint so a cold load does not block the request that triggered it.
