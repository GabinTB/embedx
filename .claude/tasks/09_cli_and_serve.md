# Task 09 — CLI and serve

## Scope

Wire the real engine to the CLI so `embedx serve` runs the server.

## Do

- `src/embedx/cli.py`:
  - `embedx serve` — build Settings, discover+rank devices, build one HFBackend
    per device (or fake/CPU per config), construct Engine, launch uvicorn with
    the FastAPI app. Log effective bind + api-key + device table at startup.
  - `embedx info` — print resolved config, devices, budgets, pooling (no serve).
  - `embedx check` — validate config + device availability, exit non-zero on
    problems (useful for systemd preflight).

## Files

- `src/embedx/cli.py`
- `tests/test_cli.py`

## Tests to add

- `embedx info`/`check` with fake/CPU devices via env; assert output + exit codes.
- `serve` wiring tested via a fast startup/shutdown or by asserting app factory
  construction (do not bind a real port in unit tests; test the factory).

## Gate

Full gate green. Commit: `feat(cli): serve/info/check commands wired to engine`.
