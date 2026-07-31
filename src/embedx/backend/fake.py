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

    `latency_s` sleeps per item to simulate slow/fast devices in
    scheduling tests.
    """

    def __init__(self, dim: int = 8, latency_s: float = 0.0) -> None:
        self.dim = dim
        self.latency_s = latency_s

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        return rng.standard_normal(self.dim, dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if self.latency_s > 0:
                time.sleep(self.latency_s)
            out[i] = self._vector(text)
        return out
