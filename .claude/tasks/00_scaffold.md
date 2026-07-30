# Task 00 — Scaffold

## Scope

Stand up the project so it installs, lints, and tests green with no real logic.

## Do

- Initialize `uv` project (`uv init` already reflected in `pyproject.toml`).
- Confirm `src/` layout with `src/embedx/__init__.py` exposing `__version__`.
- `pyproject.toml`: metadata, `hatchling` backend, `[project.optional-dependencies]`
  `gpu` and `dev`, ruff + pytest config (already provided — verify and keep).
- Add `embedx/__init__.py` with `__version__ = "0.1.0"`.
- Add a trivial `embedx/cli.py` with a `typer` app exposing `embedx --version`
  and a stub `embedx serve` that prints "not implemented yet" and exits 0.
- Add `tests/test_smoke.py`: import `embedx`, assert `__version__` is a str;
  invoke the CLI `--version` via `typer.testing.CliRunner`.
- Add `.gitignore` (Python, uv, venv, caches, notebooks checkpoints).
- Add `LICENSE` (Apache-2.0).

## Files

- `pyproject.toml` (verify)
- `src/embedx/__init__.py`
- `src/embedx/cli.py`
- `tests/test_smoke.py`
- `.gitignore`, `LICENSE`

## Tests to add

- `tests/test_smoke.py` — version import + CLI `--version`.

## Gate

`uv sync --extra dev` then:
```
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not gpu"
```
All green. Commit: `chore: scaffold uv project, packaging, ruff/pytest, CLI stub`.
