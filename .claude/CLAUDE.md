# CLAUDE.md — instructions for the coding agent

You are implementing **embedx**: a heterogeneous multi-GPU text embedding
server. One embedding model replicated across GPUs of different speed and
memory, with the *inputs* sharded between them at runtime and no configured
load ratio. Many models, loaded on demand and named per request.

Read these in order before writing any code:

1. This file. The rules below are the non-negotiable constraints, and they are
   stated here in full — there is no separate design document.
2. The specific task file under `.claude/tasks/` you are executing, and
   `.claude/tasks/README.md` for where it sits in the sequence.
3. The module you are about to change. Every core module carries a docstring
   explaining *why* it is shaped the way it is, and the comments are load
   bearing: several record measurements or production failures that a
   plausible-looking refactor would undo. `CHANGELOG.md` is the other honest
   record — it names the breaking changes and the bugs found in production.

Earlier revisions of this file pointed at a `.claude/skills/` directory that
never existed. Nothing was lost with it; the constraints below are the whole
of it.

## Rules

- **One task = one commit.** Follow `.claude/tasks/README.md` in order. Do not batch
  tasks. Do not defer tests.
- **The gate must be green on every commit:**
  ```
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy src/embedx
  uv run pytest -m "not gpu"
  ```
  `make gate` runs the same four. Never weaken, skip, or loosen a tolerance on
  a test to make it pass.
- **Core stays importable without torch.** Heavy backends (torch/transformers)
  live behind the `gpu` extra and are imported lazily inside `backend/hf.py` and
  the worker path only. A CPU CI must pass without the `gpu` extra installed.
- **Test the hard logic on CPU** via the `FakeBackend` seam. GPU-only tests are
  marked `@pytest.mark.gpu` and skipped in CPU CI.
- **Ordering invariant is sacred:** output order always equals input order,
  regardless of GPU sharding. It has dedicated tests.
- **Pooling is explicit and logged.** A wrong pooling silently produces garbage
  vectors; never guess silently.
- If the spec is ambiguous, implement the simplest correct behavior and leave a
  `# SPEC-GAP:` comment rather than inventing divergent behavior.

## Environment

- Package manager: `uv`. Install dev env: `uv sync --extra dev`. Add GPU backend:
  `uv sync --extra dev --extra gpu` (on a CUDA host).
- Python 3.11+.

## Where things go

```
src/embedx/
  config.py        Settings: holds and validates values, nothing else. No torch.
  registry.py      On-demand multi-model registry: placement, refcounts, TTL eviction.
  cli.py           serve / info / check.
  engine/
    batching.py    Length-sorted token-budget batching.
    scheduling.py  Converging work-stealing scheduler over one sorted queue.
    engine.py      Multi-worker engine + result reassembly into input order.
  gpu/
    vendor.py      The vendor seam: every CUDA-specific call behind one Protocol.
    discovery.py   Device discovery and static ranking, over that seam.
    budgets.py     Per-device token budgets, scaled by memory.
  backend/
    base.py        The backend Protocol — the seam core logic is tested against.
    fake.py        Deterministic CPU fake. Where the hard logic gets tested.
    hf.py          The real transformers/CUDA backend. The `gpu` extra lives here.
    factory.py     Backend construction, shared by whatever assembles an Engine.
  api/
    schemas.py     OpenAI-compatible bodies, plus the TEI-style /embed.
    app.py         FastAPI app factory; the model source is injected so tests fake it.
    errors.py      One OpenAI-shaped error envelope for every failure path.
  service/         The systemd unit, shipped as package data.
```

**Do not put torch imports in core modules.** `backend/hf.py` and the worker
path are the only places torch may appear, and only inside functions.
`tests/test_gpu.py` carries an ast guard that enforces both this and the rule
that nothing outside `gpu/vendor.py` touches `torch.cuda`; without the guard
the seam rots within a couple of commits.

`tests/` mostly mirrors this layout one file per module, plus the guards that
have no module of their own (`test_docker.py`, `test_readme.py`,
`test_service.py`, `test_smoke.py`). `dev/` holds the validation
notebooks and their committed outputs — not part of the package, and the
source of every benchmark number in the README. A figure that is not in
`dev/output/*.json` was not measured.
