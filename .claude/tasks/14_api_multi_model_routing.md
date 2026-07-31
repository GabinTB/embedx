# Task 14 — Route the API through the model registry

## Scope

Wire `src/embedx/registry.py` (task 13) into the HTTP layer. `model` in a
request becomes a live routing key that can trigger a load, not just an
echoed string.

## Do

- `/v1/embeddings` and `/embed`: resolve `model` through
  `ModelRegistry.get_or_load(...)`, still dispatched via `run_in_threadpool`
  as today, since a cold load is blocking and potentially slow.
- New optional request fields: `pooling`, `dtype`, `max_seq_len`,
  `keep_alive`. Document in the OpenAPI schema description, plainly and up
  front, that these only take effect on a model's *first* load — a later
  request naming an already-loaded model with a different value doesn't
  silently change it.
- Pooling conflict (task 13's rule) surfaces as **409**, not 400 — this is
  a state conflict against an already-loaded model, not a malformed request.
  Standard error envelope.
- A load failure (OOM on every device, disallowed weight format, bad hub id
  or path) returns a clean error in the envelope. Never leak a raw
  traceback for a load failure any more than for an inference failure.
- `/info` becomes per-model: list every currently resident model with its
  resolved pooling/dtype, its devices, its idle time, and the server-wide
  default `keep_alive`. Zero loaded models is a normal state (empty list),
  not an error — a fresh `serve` start has nothing loaded.
- `/health` is unchanged: still touches no model state, stays open under
  auth, stays fast.
- State plainly in a docstring or the OpenAPI description that a request
  triggering a cold load blocks until the load completes; there is no
  separate async "pull" endpoint in this phase.

## Files

`src/embedx/api/app.py`, `src/embedx/api/schemas.py`

## Tests to add

- A request for a new model triggers a load (stubbed registry) and returns
  correct results.
- A request for an already-loaded model does not trigger a second load.
- Pooling conflict returns 409 in the standard envelope shape.
- A load failure returns the standard envelope, no traceback in the body.
- `/info` reflects zero, one, and multiple loaded models correctly.
- `keep_alive` from the request is passed through to the registry call
  (assert on a stub, not on real eviction timing).

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `feat(api): route /v1/embeddings and /embed through the model registry`
