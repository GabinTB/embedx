# Task 07 — Engine wiring + OpenAI-compatible API (fake backend)

## Scope

Assemble the Engine and the HTTP layer, tested end-to-end on CPU with fake
backends. No torch.

## Do

- `src/embedx/engine/engine.py`:
  - `Engine(backends: list[EmbeddingBackend], devices: list[DeviceInfo], settings)`
    holding one backend per worker (fake in tests), the scheduler, and the
    thread pool.
  - `embed(texts: list[str]) -> np.ndarray`: sort by length, run scheduler +
    workers (each worker: claim → make_batches → backend.embed → write to output
    at original indices), return output in original order.
  - Enforce the ordering invariant by construction.
- `src/embedx/api/schemas.py`: OpenAI-compatible request/response models.
- `src/embedx/api/app.py`: FastAPI app factory; routes `/v1/embeddings`,
  `/embed`, `/health`, `/info`; bearer-auth dependency; size-limit guards;
  `run_in_threadpool` for the engine call; lifespan builds a single engine.
- `src/embedx/api/errors.py`: error handlers/response shape.

## Files

- `src/embedx/engine/engine.py`
- `src/embedx/api/__init__.py`
- `src/embedx/api/schemas.py`
- `src/embedx/api/app.py`
- `src/embedx/api/errors.py`
- `tests/test_engine.py`
- `tests/test_api.py`

## Tests to add

- Engine: output equals single-threaded reference for fake backends; with a fast
  and a slow fake backend, the faster processed strictly more items (± tolerance).
- API (TestClient + fake engine): OpenAI response shape, `index` correctness,
  `/embed`, `/health`, `/info`, auth on/off, size-limit 4xx.

## Gate

Full gate green. Commit:
`feat: engine orchestration + OpenAI-compatible FastAPI server (fake backend)`.
