# CLAUDE.md — instructions for the coding agent

You are implementing **embedx**. Read these in order before writing any code:

1. `.claude/skills/embedx-project-coding/SKILL.md` — the authoritative design and
   the non-negotiable engineering constraints.
2. The peripheral document under `.claude/skills/embedx-project-coding/` for the
   subsystem you are about to build (architecture, batching, scheduling, gpu,
   api, config, deployment, testing).
3. The specific task file under `.claude/tasks/` you are executing.

## Rules

- **One task = one commit.** Follow `.claude/tasks/README.md` in order. Do not batch
  tasks. Do not defer tests.
- **The gate must be green on every commit:**
  ```
  uv run ruff check .
  uv run ruff format --check .
  uv run pytest -m "not gpu"
  ```
  plus `uv run mypy src/embedx` once task 02 introduces it.
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

See `.claude/skills/embedx-project-coding/architecture.md` for the module layout.
Do not put torch imports in core modules.
