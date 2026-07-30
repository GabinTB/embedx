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

Build in three arcs:

1. **Skeleton & tooling** (tasks 00–02): repo runs, lints, tests, CLI stub. No
   real logic yet, just green scaffolding and the testing seam.
2. **Core logic, backend-agnostic** (tasks 03–07): batching, scheduling, engine,
   config, API — all tested on CPU with the fake backend. This is where the novel
   algorithms live and get hardened. No torch required.
3. **Real backend, service, polish** (tasks 08–12): the HF/CUDA backend behind the
   `gpu` extra, GPU tests, systemd, docs, notebooks, and the convergence of
   everything into a shippable, PyPI-publishable package.

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

Later (post-1.0, not scheduled here): base64 encoding_format, reranker endpoint,
Prometheus metrics, dynamic batching across concurrent requests, multi-model
serving.
