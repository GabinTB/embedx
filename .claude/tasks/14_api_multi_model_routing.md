# Task 14 — Route the API through the model registry

## Scope

Wire `src/embedx/registry.py` (task 13) into the HTTP layer. `model` in a
request becomes a live routing key that can trigger a load, not just an
echoed string.

**Use `Registry.acquire()`, not `get_or_load()`.** `get_or_load` does not
increment `ref_count`, so a model fetched through it can be evicted by the
reaper while a request is still using it — the exact race task 13 was built
to close. `acquire()` is the context-manager path that holds a reference for
the duration of the call:

    with registry.acquire(model_id, pooling=..., dtype=..., max_seq_len=..., keep_alive=...) as engine:
        vectors = engine.embed(texts)

The `with` block must wrap the actual `engine.embed(...)` call, not just the
lookup — the reference has to still be held while inference is running or the
whole point of reference counting is lost.

## Do

- `/v1/embeddings` and `/embed` resolve `model` through
  `Registry.acquire(...)`, wrapping the embed call, still dispatched via
  `run_in_threadpool` as today, since a cold load is blocking and potentially
  slow.
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
  default `keep_alive`. Use `Registry.list_loaded()` for this — it does not
  acquire a reference, which is correct here since listing must not keep a
  model alive. Zero loaded models is a normal state (empty list), not an
  error — a fresh `serve` start has nothing loaded.
- `/health` is unchanged: still touches no model state, stays open under
  auth, stays fast.
- State plainly in a docstring or the OpenAPI description that a request
  triggering a cold load blocks until the load completes; there is no
  separate async "pull" endpoint in this phase.

## Files

`src/embedx/api/app.py`, `src/embedx/api/schemas.py`

## Tests to add

- A request for a new model triggers a load (via `acquire()`, stubbed
  registry) and returns correct results.
- A request for an already-loaded model does not trigger a second load.
- The registry's reference is held for the duration of the embed call and
  released afterward even on failure — assert with a stub that records
  enter/exit calls, including when the engine raises mid-call.
- Pooling conflict returns 409 in the standard envelope shape.
- A load failure returns the standard envelope, no traceback in the body.
- `/info` reflects zero, one, and multiple loaded models correctly, and its
  own call does not change any model's ref_count.
- `keep_alive` from the request is passed through to `acquire()` (assert on
  a stub, not on real eviction timing).

## Gate

`uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/embedx`, `uv run pytest -m "not gpu"`.

Commit: `feat(api): route /v1/embeddings and /embed through the model registry`
