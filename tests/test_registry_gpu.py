"""GPU regression tests for VRAM release on eviction and on failed loads.

Gpu-marked and skipped in CPU CI. The companion CPU tests in
`test_registry.py` check the reference GRAPH — who is still holding a
backend when `empty_cache` runs — which is testable with stubs. These check
the only thing that actually matters to an operator, which is not testable
with stubs at all: whether the bytes come back.

The bug these exist for shipped in 0.3.0. `/info` reported no models while
`nvidia-smi` reported the full weight of an evicted 4B model still held by
the server process, indefinitely, with nothing in the log. It survived a
suite of eviction tests because those tests assert that a teardown hook ran
— and it did. Nothing measured memory, because nothing could: a stub
backend has no weights, so releasing it correctly and releasing it
incorrectly look identical.

Every assertion here is scaled to the model's OWN measured size rather than
to a fixed byte count, so a different checkpoint or a different dtype does
not silently turn the tolerance into a pass-everything.
"""

from __future__ import annotations

from typing import Any

import pytest

from embedx.config import Pooling, Settings
from embedx.registry import ModelRegistry

pytestmark = pytest.mark.gpu

# Small and already used by the rest of the GPU suite: this measures the
# release path, and a 4B model would measure download bandwidth.
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# What counts as "returned". The leak strands a whole model copy, so any
# threshold well under one copy separates the two states; 5% leaves room
# for allocator bookkeeping without leaving room for a model.
LEAK_FRACTION = 0.05


@pytest.fixture(scope="module")
def torch() -> Any:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    return torch


@pytest.fixture(scope="module")
def devices(torch: Any) -> list[Any]:
    from embedx.gpu.discovery import discover_devices, rank_devices

    return rank_devices(discover_devices(), {})


def allocated(torch: Any, devices: list[Any]) -> dict[int, int]:
    return {device.index: int(torch.cuda.memory_allocated(device.index)) for device in devices}


def reserved(torch: Any, devices: list[Any]) -> dict[int, int]:
    """What the process holds from the driver — i.e. what nvidia-smi shows.

    The distinction is the whole reason the production symptom was
    confusing. `memory_allocated` can fall to zero while `memory_reserved`
    stays high: the allocator has the blocks cached and only `empty_cache`
    hands them back. A fix that frees the tensors but still calls
    `empty_cache` too early looks clean in one metric and unchanged in the
    other, so both are asserted.
    """
    return {device.index: int(torch.cuda.memory_reserved(device.index)) for device in devices}


def load_and_evict(devices: list[Any], model: str = MODEL, **overrides: Any) -> None:
    """One complete cycle, holding nothing afterwards.

    `keep_alive=0` evicts when the acquire block exits, which is the path a
    request takes rather than the reaper's — same `_evict`, no sleeping.
    """
    registry = ModelRegistry(devices, settings=Settings(**overrides))
    with registry.acquire(model, pooling=Pooling.MEAN, keep_alive=0.0) as engine:
        engine.embed(["a short sentence", "another one, slightly longer than the first"])


@pytest.fixture(scope="module")
def warm(torch: Any, devices: list[Any]) -> None:
    """Pay the one-off, non-model device costs before anything is measured.

    torch allocates a cuBLAS workspace per device on the first matmul and
    keeps it for the life of the process — 32 MiB on the card this was
    written against. It is not a leak and no `empty_cache` returns it, but
    it is indistinguishable from one if the baseline is taken before the
    first forward pass rather than after it.
    """
    load_and_evict(devices)


def test_evicting_a_model_returns_its_device_memory(
    torch: Any, devices: list[Any], warm: None
) -> None:
    """The acceptance test: allocated bytes come back to the baseline.

    Fails on 0.3.0. `Engine._length_fn` is a bound method of `backends[0]`,
    so the first device's model stayed reachable through eviction; the
    release loop's own variable held the last one; and a transformer is
    cyclic, so even unreachable it needed a collection before the allocator
    would count it free. `empty_cache` returns only free blocks, so it
    returned nothing and said nothing.
    """
    baseline = allocated(torch, devices)
    baseline_reserved = reserved(torch, devices)

    registry = ModelRegistry(devices, settings=Settings())
    with registry.acquire(MODEL, pooling=Pooling.MEAN, keep_alive=0.0) as engine:
        engine.embed(["a short sentence", "another one, slightly longer than the first"])
        peak = allocated(torch, devices)

    assert registry.list_loaded() == []
    after = allocated(torch, devices)
    after_reserved = reserved(torch, devices)

    weights = {index: peak[index] - baseline[index] for index in baseline}
    assert any(size > 0 for size in weights.values()), (
        "the model never reached a device; this test would pass vacuously"
    )

    for index, size in weights.items():
        if size <= 0:
            continue  # nothing was placed here, nothing to give back
        held = after[index] - baseline[index]
        assert held < size * LEAK_FRACTION, (
            f"device {index} still holds {held / 2**20:.1f} MiB of the "
            f"{size / 2**20:.1f} MiB model after eviction"
        )
        held_reserved = after_reserved[index] - baseline_reserved[index]
        assert held_reserved < size * LEAK_FRACTION, (
            f"device {index} released the tensors but never handed the "
            f"{held_reserved / 2**20:.1f} MiB back to the driver: this is what "
            "nvidia-smi keeps showing"
        )


def test_repeated_load_evict_cycles_do_not_grow(torch: Any, devices: list[Any], warm: None) -> None:
    """A leak of a fraction of a model per cycle still fills the card.

    The single-cycle test above uses a tolerance; this one has none to
    spend, because anything that is not returned accumulates. Three cycles
    must land on exactly the same number as one.
    """
    baseline = allocated(torch, devices)
    for _ in range(3):
        load_and_evict(devices)
    assert allocated(torch, devices) == baseline


def test_a_load_that_fails_after_placing_weights_frees_them(
    torch: Any, devices: list[Any], warm: None
) -> None:
    """The load-failure leak, which is worse than the eviction one.

    A model half-placed when the load aborts is referenced only by locals
    of a frame that is unwinding, and by the traceback of the exception
    carrying it — so no caller ever sees it to free it, and a client
    retrying its request walks the card out of memory a model at a time.

    The OOM is injected rather than provoked: raising it from a frame that
    holds a fully constructed backend reproduces the property under test —
    device memory reachable only through `exc.__traceback__` at the moment
    `empty_cache` runs — deterministically. Provoking a real one means
    tuning a memory cap to fail part way through one specific checkpoint on
    one specific card, which tests the tuning more than the code.
    """
    from embedx.backend.factory import hf_backend_factory

    baseline = allocated(torch, devices)
    placed: list[int] = []

    def oom_after_placing(**kwargs: Any) -> Any:
        # Unused on purpose, and the `noqa` is load-bearing: `backend` being
        # nothing but a local of a frame that lands in a traceback is the
        # entire condition under test. Deleting it deletes the test.
        backend = hf_backend_factory(**kwargs)  # noqa: F841
        placed.append(int(kwargs["device_index"]))
        raise torch.cuda.OutOfMemoryError(f"injected after placing on {kwargs['device_index']}")

    registry = ModelRegistry(devices, settings=Settings(), backend_factory=oom_after_placing)
    with pytest.raises(Exception) as excinfo:
        registry.get_or_load(MODEL, pooling=Pooling.MEAN)

    assert "did not fit on any" in str(excinfo.value)
    assert placed == [device.index for device in devices], "every device should have been tried"
    assert registry.list_loaded() == []
    assert allocated(torch, devices) == baseline, (
        "a failed load left weights on the card; retries would exhaust it"
    )
