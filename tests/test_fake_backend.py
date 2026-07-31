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


def test_latency_honored() -> None:
    backend = FakeBackend(dim=4, latency_s=0.01)
    start = time.perf_counter()
    backend.embed(["a", "b", "c", "d", "e"])
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.05


def test_zero_latency_is_fast() -> None:
    backend = FakeBackend(dim=4)
    start = time.perf_counter()
    backend.embed(["a"] * 100)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
