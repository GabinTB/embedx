"""Deterministic fake backend for CPU-only tests.

Pure Python + numpy — no torch. Vectors are derived from a stable hash of
each text so results are reproducible across processes and runs (unlike
built-in `hash()`, which is salted per interpreter).
"""

from __future__ import annotations

import hashlib
import time

import numpy as np


class FakeBackend:
    """Embedding backend that maps each text to a stable pseudo-random vector.

    Simulated cost per item is `latency_s + latency_per_token * len(text)`:
    a constant per-item overhead plus a length-sensitive component. The
    length-sensitive part is what makes scheduling tests discriminating —
    the scheduler exists because longer texts cost more on a real GPU, and
    a flat per-item cost could not tell good batching from bad. `len(text)`
    is a deliberately cheap token proxy; this is a test double, not a
    tokenizer.

    Both latencies default to 0.0, so the backend is effectively instant
    unless a test opts in to simulated slow/fast devices.
    """

    def __init__(
        self, dim: int = 8, latency_s: float = 0.0, latency_per_token: float = 0.0
    ) -> None:
        self.dim = dim
        self.latency_s = latency_s
        self.latency_per_token = latency_per_token

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        return rng.standard_normal(self.dim, dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            delay = self.latency_s + self.latency_per_token * len(text)
            if delay > 0:
                time.sleep(delay)
            out[i] = self._vector(text)
        return out
