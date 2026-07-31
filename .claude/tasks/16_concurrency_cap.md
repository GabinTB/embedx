# Task 16 — Bound concurrent requests and concurrent model loads

## Scope

embedx currently has no cap on in-flight requests (flagged as a gap at
0.1.0: `run_in_threadpool` is called with no limit, so N concurrent requests
mean N schedulers and up to N × devices worker threads contending on
per-device locks). This gets more pressing with the registry (task 13),
since a request can now also trigger a cold model load, which is far more
expensive than a normal embed call and contends for VRAM and PCIe bandwidth
across simultaneous loads.

## Do

- `Settings.max_concurrent_requests: int` (sensible default; document the
  reasoning, e.g. scaled to device count) gates entry to the embedding
  routes: acquire before dispatching to the registry/engine, release after,
  regardless of success or failure.
- `Settings.request_queue_timeout_s`: a request that can't acquire a slot
  within this window returns **503** in the standard error envelope rather
  than queuing indefinitely.
- Loads and evictions are not gated by the same semaphore as ordinary
  requests — a request blocked in the queue behind a cold load elsewhere
  must not starve indefinitely just because the request cap is full of
  waiting-on-load callers. Give cold loads their own smaller cap,
  `Settings.max_concurrent_loads` (default small, e.g. 2 — justify it with a
  comment; revisit once task 17's benchmark exists and gives a real number
  for how much a simultaneous second load degrades an in-flight one on this
  hardware).
- `/info` reports live counts: current in-flight requests, current
  in-flight loads, and the configured caps for both — this needs to be
  observable, not a silent ceiling someone discovers via a 503.

## Files

`src/embedx/api/app.py`, `src/embedx/config.py`

## Tests to add

- More concurrent requests than `max_concurrent_requests`: excess requests
  wait and then succeed once a slot frees (stub engine with a controllable
  artificial delay).
- A request that waits past `request_queue_timeout_s` returns 503, not a
  hang — bound the test's own wall-clock time so a regression fails fast
  rather than timing out the test suite.
- `max_concurrent_loads` bounds simultaneous cold loads independently of the
  request cap (stub with a slow, controllable load).
- `/info` reports accurate live counts under concurrent load.

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `feat(api): bound concurrent requests and concurrent model loads`
