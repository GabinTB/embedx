# Task 21 — Split the PyPI distribution from the server

## Scope

PyPI ships the **library**. The server ships via Docker and from source.

Second of three sequential commits (20 audit, 21 split, 22 README). Task 20
must land first.

## The two artifacts

**PyPI — library only.** Batching, the converging scheduler, the `Engine`,
device ranking and budgets, the model registry, config. Someone doing local
multi-GPU batch work imports it and calls it directly, with no HTTP
round-trip. That use case is real: it is what a local batch job on a machine
with two GPUs actually wants, and the round-trip is pure overhead there.

**Docker and git clone — everything.** The FastAPI app, the HTTP layer, and
`embedx serve`. Docker is the verified artifact: it is where the Python
version, the torch wheel, the CUDA runtime and the C toolchain are pinned and
tested together, and it is the only path where a RoPE-family model is known
to work end to end.

## Naming

- PyPI distribution: `embedx-inference`
- Import name: `embedx`, unchanged. Docs use `import embedx as ebx` as the
  house convention, the way numpy uses `np`.
- Repo name: `embedx`, unchanged.

Distribution and import names are independent and the split is ordinary
(`scikit-learn`/`sklearn`, `beautifulsoup4`/`bs4`, `opencv-python`/`cv2`).
`embedx` is taken on PyPI by a dormant 2016 project, so the short name is not
available regardless.

## Mechanism

This is a **build-configuration exclusion, not a restructure**. One source
tree, one `pyproject.toml`. Check which build backend is declared and use its
exclusion mechanism — with hatchling that is
`[tool.hatch.build.targets.wheel] exclude`. Docker keeps installing from
source and therefore keeps getting everything.

## Report before acting

- Which modules land on which side, and any genuinely ambiguous ones.
  `registry.py` is **library**, not server: a script embedding several models
  in sequence wants on-demand loading and TTL eviction.
- Whether `fastapi`, `uvicorn` and the CLI framework are currently core
  dependencies and need moving to an optional group.
- Whether anything in the library half imports from `api/`. It should not. If
  it does, that is a real coupling to report, not to silently work around.

## `embedx serve` on a library install

Register the subcommand **conditionally**: when `src/embedx/api/` is absent,
it is simply not registered, so it does not appear in `embedx --help` and
there is no failure path to handle.

Add one line to the CLI help epilogue noting that the HTTP server ships via
Docker or from source — someone who expected `serve` and finds it missing
should learn where it went rather than concluding the project has no server.
Do not implement a runtime error for this.

## CI

The packaging job's torch-free import check is what proves the core imports
without torch. Once the wheel contents change, confirm it still proves what
it claims, and state what it should assert now. It must also verify that the
built wheel does **not** contain the API layer — an exclusion nobody checks
is an exclusion that silently stops working.

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `build: ship the library on PyPI, the server via Docker`
