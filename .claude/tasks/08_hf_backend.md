# Task 08 — Real Hugging Face / sentence-transformers backend

## Scope

Implement the real GPU backend behind the `gpu` extra. First code that imports
torch — must be lazy and isolated.

## Do

- `src/embedx/backend/hf.py`:
  - `HFBackend(model_id, device, pooling, normalize, dtype, max_seq_length)`
    implementing `EmbeddingBackend`.
  - Loads via sentence-transformers when the model is an ST model, else
    transformers `AutoModel` + explicit pooling (`mean`/`cls`/`last-token`).
  - `pooling="auto"` reads ST config / model config; the chosen pooling is
    returned/logged (never silent).
  - True token-length function exposed for the batcher (`length_fn` using the
    tokenizer).
  - All torch/transformers imports inside this module, loaded lazily.
- Wire `discover_devices()` + `HFBackend` into the engine construction path
  (a factory that builds one `HFBackend` per selected device).

## Files

- `src/embedx/backend/hf.py`
- `tests/test_hf_backend.py` (all `@pytest.mark.gpu`)

## Tests to add (gpu-marked, skipped in CPU CI)

- Loads a small known model on CUDA; `dim` correct.
- Pooling correctness: embeddings match a sentence-transformers reference for a
  known model within tolerance.
- Truncation counter increments on an over-long input.

## Gate

CPU gate green (gpu tests skipped). On a GPU host, `pytest -m gpu` green. Commit:
`feat(backend): HF/sentence-transformers CUDA backend with explicit pooling`.
