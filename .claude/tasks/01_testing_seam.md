# Task 01 — Testing seam (backend protocol + fake backend)

## Scope

Create the abstraction that lets all core logic be tested on CPU without torch.

## Do

- `src/embedx/backend/base.py`: define `EmbeddingBackend` (typing.Protocol) with
  `dim: int` and `embed(self, texts: list[str]) -> np.ndarray` returning shape
  `(len(texts), dim)`.
- `src/embedx/backend/fake.py`: `FakeBackend(dim=8, latency_s=0.0)` implementing
  the protocol deterministically: each text maps to a stable vector derived from
  a hash (reproducible), optionally sleeping `latency_s` per item to simulate
  slow/fast devices. Must be pure-Python + numpy, no torch.
- `src/embedx/backend/__init__.py`: re-export `EmbeddingBackend`, `FakeBackend`.

## Files

- `src/embedx/backend/base.py`
- `src/embedx/backend/fake.py`
- `src/embedx/backend/__init__.py`
- `tests/test_fake_backend.py`

## Tests to add

- Determinism: same text → identical vector across calls.
- Shape: `embed(n texts).shape == (n, dim)`.
- Distinctness: different texts → different vectors (with high probability).
- Latency honored (approx) when set.

## Gate

Standard gate green. Commit: `feat(backend): add EmbeddingBackend protocol and deterministic FakeBackend`.
