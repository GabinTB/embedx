"""Tests for the multi-worker Engine (task 07a)."""

from __future__ import annotations

import random
import threading
import time

import numpy as np
import pytest

from embedx.backend import FakeBackend
from embedx.config import Settings
from embedx.engine import Engine
from embedx.gpu.discovery import DeviceInfo

GIB = 2**30


def make_settings(**overrides: object) -> Settings:
    kwargs: dict[str, object] = {"model_id": "test-model", "pooling": "mean"}
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def make_device(index: int) -> DeviceInfo:
    return DeviceInfo(
        index=index,
        name=f"Fake GPU {index}",
        total_memory_bytes=16 * GIB,
        multi_processor_count=100,
        capability=(8, 0),
    )


def make_engine(backends: list, **settings_overrides: object) -> Engine:
    devices = [make_device(i) for i in range(len(backends))]
    return Engine(backends, devices, make_settings(**settings_overrides))


class RecordingBackend:
    """Counts items and batches; delegates to an inner FakeBackend."""

    def __init__(self, inner: FakeBackend) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.items = 0
        self.batches = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.items += len(texts)
        self.batches += 1
        return self.inner.embed(texts)


class ReentrancyGuardBackend:
    """Raises if two threads are ever inside embed() at once."""

    def __init__(self, inner: FakeBackend) -> None:
        self.inner = inner
        self.dim = inner.dim
        self._entered = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._entered.acquire(blocking=False):
            raise RuntimeError("backend entered concurrently")
        try:
            time.sleep(0.001)  # widen the race window
            return self.inner.embed(texts)
        finally:
            self._entered.release()


class ExplodingBackend:
    dim = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        if any("BOOM" in text for text in texts):
            raise RuntimeError("kaboom")
        return np.zeros((len(texts), self.dim), dtype=np.float32)


class InconsistentDimBackend:
    """Returns a different width on the second call."""

    dim = 4

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        width = 4 if self.calls == 1 else 8
        return np.zeros((len(texts), width), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Ordering invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_workers", [1, 2, 3, 4])
def test_ordering_invariant_randomized(n_workers: int) -> None:
    rng = random.Random(n_workers)
    reference_backend = FakeBackend(dim=8)
    for _ in range(5):
        texts = [
            "".join(rng.choices("abc ", k=rng.randint(0, 30))) for _ in range(rng.randint(1, 80))
        ]
        # Edge cases: duplicates, empty strings, one text far over any budget.
        # (device_budgets floors budgets at min_tokens=512, so the effective
        # budget is 600 and the long text must exceed that.)
        texts += ["dup", "dup", "", "z" * 2000]
        rng.shuffle(texts)
        reference = reference_backend.embed(texts)

        engine = make_engine([FakeBackend(dim=8) for _ in range(n_workers)], max_batch_tokens=600)
        np.testing.assert_array_equal(engine.embed(texts), reference)


# --------------------------------------------------------------------------- #
# Load split
# --------------------------------------------------------------------------- #


def test_fast_worker_processes_strictly_more_items() -> None:
    # 10x speed gap via the length-sensitive cost model; items per worker,
    # not wall clock, is what we assert on. Budget 512 over length-50 texts
    # gives 10-item batches: 30 claims for the two workers to split.
    fast = RecordingBackend(FakeBackend(dim=4, latency_per_token=2e-5))
    slow = RecordingBackend(FakeBackend(dim=4, latency_per_token=2e-4))
    engine = make_engine([fast, slow], max_batch_tokens=512)

    texts = ["x" * 50] * 300
    engine.embed(texts)

    assert fast.items + slow.items == 300
    assert fast.items > slow.items


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def test_worker_exception_carries_device_index() -> None:
    engine = Engine([ExplodingBackend()], [make_device(7)], make_settings(max_batch_tokens=16))
    with pytest.raises(RuntimeError, match=r"device 7.*kaboom"):
        engine.embed(["ok", "BOOM", "ok"])


def test_worker_exception_with_multiple_workers() -> None:
    engine = make_engine([ExplodingBackend(), ExplodingBackend()], max_batch_tokens=16)
    with pytest.raises(RuntimeError, match=r"embedding failed on device \d+"):
        engine.embed(["BOOM" + str(i) for i in range(20)])


def test_inconsistent_backend_dimension_raises() -> None:
    # Budget 600 over length-50 texts forces three batches (12 items each)
    # from the single backend, so the width flips on the second call.
    engine = make_engine([InconsistentDimBackend()], max_batch_tokens=600)
    with pytest.raises(ValueError, match="dimension"):
        engine.embed(["a" * 50] * 30)


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_embed_calls_correct_and_never_reenter_backends() -> None:
    engine = make_engine(
        [ReentrancyGuardBackend(FakeBackend(dim=8)) for _ in range(2)], max_batch_tokens=32
    )
    rng = random.Random(9)
    inputs = [
        ["".join(rng.choices("abcdef", k=rng.randint(1, 20))) for _ in range(40)] for _ in range(4)
    ]
    reference_backend = FakeBackend(dim=8)
    references = [reference_backend.embed(texts) for texts in inputs]

    outputs: list[np.ndarray | None] = [None] * len(inputs)
    errors: list[Exception] = []

    def call(i: int) -> None:
        try:
            outputs[i] = engine.embed(inputs[i])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(inputs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors  # a reentered backend would have raised
    for output, reference in zip(outputs, references, strict=True):
        assert output is not None
        np.testing.assert_array_equal(output, reference)


# --------------------------------------------------------------------------- #
# Edges and validation
# --------------------------------------------------------------------------- #


def test_empty_input_returns_empty_array_and_starts_no_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine([FakeBackend(dim=8)])

    def _no_threads(*args: object, **kwargs: object) -> None:
        pytest.fail("embed([]) must not start any thread")

    monkeypatch.setattr(threading, "Thread", _no_threads)
    out = engine.embed([])
    assert out.shape[0] == 0
    assert out.dtype == np.float32


def test_engine_uses_injected_length_fn() -> None:
    # Every text measures far over any budget, so each must form a batch of
    # one — proof the scheduler is packing with the injected length, not len.
    backend = RecordingBackend(FakeBackend(dim=4))
    engine = Engine([backend], [make_device(0)], make_settings(), length_fn=lambda _text: 10**9)
    out = engine.embed(["a", "b", "c"])
    assert out.shape == (3, 4)
    assert backend.batches == 3


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="at least one backend"):
        Engine([], [], make_settings())
    with pytest.raises(ValueError, match="equal length"):
        Engine([FakeBackend()], [make_device(0), make_device(1)], make_settings())
