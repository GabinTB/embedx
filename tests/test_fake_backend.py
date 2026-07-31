"""Tests for the FakeBackend testing seam (task 01)."""

from __future__ import annotations

import time

import numpy as np

from embedx.backend import EmbeddingBackend, FakeBackend


def test_satisfies_protocol() -> None:
    assert isinstance(FakeBackend(), EmbeddingBackend)


def test_determinism_across_calls_and_instances() -> None:
    a = FakeBackend(dim=8).embed(["hello", "world"])
    b = FakeBackend(dim=8).embed(["hello", "world"])
    np.testing.assert_array_equal(a, b)


def test_shape() -> None:
    backend = FakeBackend(dim=16)
    out = backend.embed(["a", "b", "c"])
    assert out.shape == (3, 16)
    assert backend.embed([]).shape == (0, 16)


def test_distinct_texts_give_distinct_vectors() -> None:
    out = FakeBackend(dim=8).embed(["alpha", "beta", "gamma"])
    assert not np.array_equal(out[0], out[1])
    assert not np.array_equal(out[0], out[2])
    assert not np.array_equal(out[1], out[2])


def _timed(backend: FakeBackend, texts: list[str]) -> float:
    start = time.perf_counter()
    backend.embed(texts)
    return time.perf_counter() - start


def test_constant_latency_honored() -> None:
    # sleep() guarantees at least the requested duration, so the lower
    # bound is exact and load-independent: 5 items x 0.01s >= 0.05s.
    elapsed = _timed(FakeBackend(dim=4, latency_s=0.01), ["a", "b", "c", "d", "e"])
    assert elapsed >= 0.05


def test_per_token_latency_scales_with_length() -> None:
    backend = FakeBackend(dim=4, latency_per_token=0.001)
    short = _timed(backend, ["a" * 10])  # >= 0.01s
    long = _timed(backend, ["a" * 100])  # >= 0.10s
    assert short >= 0.01
    assert long >= 0.10
    # 10x the length must cost measurably more (loose ordering, not a ratio,
    # to stay robust on loaded runners).
    assert long > short


def test_constant_and_per_token_latencies_are_additive() -> None:
    # Per item: 0.02 + 0.001 * 30 = 0.05s; 3 items >= 0.15s. Either
    # component alone would only guarantee 0.06s / 0.09s, so this lower
    # bound is only reachable if both are applied.
    backend = FakeBackend(dim=4, latency_s=0.02, latency_per_token=0.001)
    elapsed = _timed(backend, ["a" * 30] * 3)
    assert elapsed >= 0.15


def test_default_latencies_are_fast() -> None:
    # The zero-latency path (both defaults 0.0) must be at least an order
    # of magnitude faster for 100 items than a single item with a real
    # latency — i.e. hashing dominates nothing.
    default_elapsed = _timed(FakeBackend(dim=4), ["a"] * 100)
    latency_elapsed = _timed(FakeBackend(dim=4, latency_s=0.25), ["a"])
    assert default_elapsed * 10 < latency_elapsed
