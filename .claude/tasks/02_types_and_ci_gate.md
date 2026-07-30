# Task 02 — Types and CI gate

## Scope

Lock in the quality gate so every later task is enforced automatically.

## Do

- Add `mypy` config (already in `pyproject.toml`); ensure `mypy src/embedx`
  passes on current code (annotate as needed).
- Add GitHub Actions workflow `.github/workflows/ci.yml`:
  - matrix Python 3.11, 3.12
  - `uv sync --extra dev`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run mypy src/embedx`
  - `uv run pytest -m "not gpu"`
- Add a `Makefile` or `justfile` (optional) with `lint`, `fmt`, `test`, `gate`
  targets mirroring the gate for local use.

## Files

- `.github/workflows/ci.yml`
- (optional) `Makefile`
- annotations as needed in existing files

## Tests to add

- None new; this task makes the gate authoritative.

## Gate

Full gate incl. `mypy src/embedx` green locally and in CI. Commit:
`ci: enforce ruff, mypy, pytest gate on 3.11/3.12`.
