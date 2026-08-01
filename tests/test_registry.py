"""Tests for the on-demand model registry (task 13).

No real models and no torch: `HFBackend` construction is replaced wholesale
by an injected factory, so every path here — locking, placement, pooling
conflicts, eviction — runs on CPU CI.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import pytest

from embedx.backend import FakeBackend
from embedx.config import Dtype, Pooling
from embedx.gpu.discovery import DeviceInfo
from embedx.registry import (
    DEFAULT_KEEP_ALIVE_S,
    ModelPlacementError,
    ModelRegistry,
    PoolingConflictError,
    PoolingRequiredError,
    RegistryError,
    UnsupportedWeightFormatError,
)

GIB = 2**30


class OutOfMemoryError(Exception):
    """Stands in for torch.cuda.OutOfMemoryError, which CPU CI cannot import.

    The registry matches CUDA OOM by type name as well as by isinstance
    precisely so this seam exists, which is why the name here has to be
    exactly torch's: anything else lands in the re-raise branch instead.
    """


def make_devices(count: int = 2) -> list[DeviceInfo]:
    return [
        DeviceInfo(
            index=index,
            name=f"Fake GPU {index}",
            total_memory_bytes=16 * GIB,
            multi_processor_count=100 - index,
            capability=(8, 0),
            score=160.0 - index,
            weight=1.0 - 0.1 * index,
        )
        for index in range(count)
    ]


class StubBackend:
    """A backend double that records its device and can be torn down."""

    def __init__(self, model_id: str, device_index: int, pooling: Pooling) -> None:
        self.model_id = model_id
        self.device_index = device_index
        self.pooling = pooling
        self.closed = False
        self._inner = FakeBackend(dim=4)
        self.dim = self._inner.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._inner.embed(texts)

    def length_fn(self, text: str) -> int:
        return len(text)

    def close(self) -> None:
        self.closed = True


class StubFactory:
    """Injectable `BackendFactory` with a load counter and failure control.

    `oom_on` is the set of device indices that raise OOM instead of
    returning a backend; `delay_s` makes a load slow enough for concurrency
    races to be real rather than theoretical.
    """

    def __init__(
        self,
        oom_on: set[int] | None = None,
        delay_s: float = 0.0,
        fail_with: Exception | None = None,
    ) -> None:
        self.oom_on = oom_on or set()
        self.delay_s = delay_s
        self.fail_with = fail_with
        self.loads = 0  # backends constructed
        self.model_loads: list[str] = []  # one entry per load episode
        self.created: list[StubBackend] = []
        self._episode_devices: dict[str, set[int]] = {}
        self._lock = threading.Lock()

    def __call__(
        self,
        *,
        model_id: str,
        device_index: int,
        pooling: Pooling,
        normalize: bool,
        dtype: Dtype,
        max_seq_length: int | None,
        revision: str | None,
        length_cache: Any,
    ) -> StubBackend:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with
        if device_index in self.oom_on:
            raise OutOfMemoryError(f"CUDA out of memory on {device_index}")
        with self._lock:
            self.loads += 1
            # One "load episode" is the registry walking the device list
            # once for a model. Each device is visited at most once per
            # episode, so seeing a repeat means a fresh load (a reload after
            # eviction, or a duplicate the per-model lock should have
            # prevented).
            seen = self._episode_devices.setdefault(model_id, set())
            if device_index in seen:
                seen.clear()
            if not seen:
                self.model_loads.append(model_id)
            seen.add(device_index)
            backend = StubBackend(model_id, device_index, pooling)
            self.created.append(backend)
        return backend


def safetensors_listing(model_id: str, revision: str | None) -> list[str]:
    return ["config.json", "model.safetensors", "tokenizer.json"]


def make_registry(
    factory: StubFactory | None = None,
    devices: int = 2,
    lister: Any = safetensors_listing,
) -> tuple[ModelRegistry, StubFactory]:
    factory = factory or StubFactory()
    registry = ModelRegistry(
        make_devices(devices),
        reaper_interval_s=0.01,
        backend_factory=factory,
        weight_file_lister=lister,
    )
    return registry, factory


# --------------------------------------------------------------------------- #
# Loading and reuse
# --------------------------------------------------------------------------- #


def test_first_load_returns_a_working_engine() -> None:
    registry, factory = make_registry()
    engine = registry.get_or_load("org/model", pooling=Pooling.MEAN)
    assert factory.model_loads == ["org/model"]
    assert factory.loads == 2  # one backend per device

    vectors = engine.embed(["hello", "a longer piece of text", "x"])
    assert vectors.shape == (3, 4)
    np.testing.assert_array_equal(
        vectors, FakeBackend(dim=4).embed(["hello", "a longer piece of text", "x"])
    )


def test_second_call_reuses_without_loading_again() -> None:
    registry, factory = make_registry()
    first = registry.get_or_load("org/model", pooling=Pooling.MEAN)
    second = registry.get_or_load("org/model")  # pooling=None: reuse as-is
    with registry.acquire("org/model") as third:
        assert third is first
    assert second is first
    assert factory.model_loads == ["org/model"]


def test_concurrent_first_loads_of_one_model_load_exactly_once() -> None:
    # The delay makes the race real: without the per-model lock every thread
    # would find the entry absent and load its own copy.
    registry, factory = make_registry(StubFactory(delay_s=0.05))
    engines: list[Any] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def worker() -> None:
        try:
            barrier.wait()
            with registry.acquire("org/model", pooling=Pooling.MEAN) as engine:
                engines.append(engine)
        except BaseException as exc:  # surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert factory.model_loads == ["org/model"]
    assert factory.loads == 2  # two devices, once each
    assert len(engines) == 6
    assert all(engine is engines[0] for engine in engines)
    assert registry.list_loaded()[0].ref_count == 0


def test_different_model_ids_load_in_parallel() -> None:
    # Per-model locks, not one global load lock: two first-loads of distinct
    # models must overlap, so wall time is one delay and not two.
    delay = 0.2
    registry, factory = make_registry(StubFactory(delay_s=delay))
    barrier = threading.Barrier(2)

    def worker(model_id: str) -> None:
        barrier.wait()
        with registry.acquire(model_id, pooling=Pooling.MEAN):
            pass

    threads = [threading.Thread(target=worker, args=(f"org/model-{n}",)) for n in range(2)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started

    assert sorted(factory.model_loads) == ["org/model-0", "org/model-1"]
    # Each model loads 2 backends sequentially => ~2 delays per model, and
    # the two models overlap => ~2 delays total. Serialised behind one
    # global load lock it would be ~4, so 3 separates the two cleanly.
    assert elapsed < 3 * delay, f"loads appear serialised: {elapsed:.2f}s"


def test_non_oom_load_failure_propagates_rather_than_retrying_every_device() -> None:
    registry, _ = make_registry(StubFactory(fail_with=ValueError("no such model")))
    with pytest.raises(ValueError, match="no such model"):
        registry.get_or_load("org/typo", pooling=Pooling.MEAN)
    assert registry.list_loaded() == []


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #


def test_first_load_without_pooling_is_refused() -> None:
    registry, factory = make_registry()
    with pytest.raises(PoolingRequiredError) as excinfo:
        registry.get_or_load("org/model")
    assert excinfo.value.model_id == "org/model"
    assert "org/model" in str(excinfo.value)
    assert factory.loads == 0
    assert registry.list_loaded() == []


def test_matching_pooling_on_a_later_call_is_fine() -> None:
    registry, factory = make_registry()
    first = registry.get_or_load("org/model", pooling=Pooling.CLS)
    assert registry.get_or_load("org/model", pooling=Pooling.CLS) is first
    assert factory.model_loads == ["org/model"]


def test_conflicting_pooling_raises_and_leaves_the_entry_untouched() -> None:
    registry, factory = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN)
    before = registry.list_loaded()

    with pytest.raises(PoolingConflictError) as excinfo:
        registry.get_or_load("org/model", pooling=Pooling.CLS)
    assert excinfo.value.requested is Pooling.CLS
    assert excinfo.value.resolved is Pooling.MEAN

    after = registry.list_loaded()
    assert [(s.model_id, s.pooling, s.ref_count) for s in after] == [
        (s.model_id, s.pooling, s.ref_count) for s in before
    ]
    assert factory.model_loads == ["org/model"]  # no reload
    assert after[0].pooling is Pooling.MEAN


def test_conflicting_pooling_through_acquire_does_not_leak_a_reference() -> None:
    registry, _ = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN)
    with pytest.raises(PoolingConflictError), registry.acquire("org/model", pooling=Pooling.CLS):
        pass  # pragma: no cover - acquire raises before the body runs
    assert registry.list_loaded()[0].ref_count == 0


# --------------------------------------------------------------------------- #
# Device placement
# --------------------------------------------------------------------------- #


def test_oom_on_the_fast_device_lands_the_model_on_the_slow_one() -> None:
    registry, factory = make_registry(StubFactory(oom_on={0}))
    registry.get_or_load("org/big", pooling=Pooling.MEAN)

    assert [backend.device_index for backend in factory.created] == [1]
    status = registry.list_loaded()[0]
    assert status.device_indices == (1,)


def test_oom_on_every_device_reports_every_device_tried() -> None:
    registry, _ = make_registry(StubFactory(oom_on={0, 1}))
    with pytest.raises(ModelPlacementError) as excinfo:
        registry.get_or_load("org/huge", pooling=Pooling.MEAN)

    assert [index for index, _ in excinfo.value.failures] == [0, 1]
    assert all("OutOfMemory" in reason for _, reason in excinfo.value.failures)
    assert "device 0" in str(excinfo.value) and "device 1" in str(excinfo.value)
    assert registry.list_loaded() == []


def test_placement_walks_devices_in_the_given_rank_order() -> None:
    registry, factory = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN)
    assert [backend.device_index for backend in factory.created] == [0, 1]


# --------------------------------------------------------------------------- #
# Weight format
# --------------------------------------------------------------------------- #


def test_pickle_only_weights_are_refused_before_any_backend_is_built() -> None:
    def bin_only(model_id: str, revision: str | None) -> list[str]:
        return ["config.json", "pytorch_model.bin", "tokenizer.json"]

    registry, factory = make_registry(lister=bin_only)
    with pytest.raises(UnsupportedWeightFormatError) as excinfo:
        registry.get_or_load("org/legacy", pooling=Pooling.MEAN)

    assert excinfo.value.files == ["pytorch_model.bin"]
    assert "pytorch_model.bin" in str(excinfo.value)
    assert factory.loads == 0, "the weight check must run before construction"
    assert registry.list_loaded() == []


def test_safetensors_alongside_a_legacy_file_loads_silently() -> None:
    def both(model_id: str, revision: str | None) -> list[str]:
        return ["pytorch_model.bin", "model.safetensors"]

    registry, factory = make_registry(lister=both)
    registry.get_or_load("org/both", pooling=Pooling.MEAN)
    assert factory.model_loads == ["org/both"]


def test_an_unlistable_repo_loads_anyway() -> None:
    # Fails open on purpose: a cached model behind an unreachable hub must
    # still serve. The warning is the record.
    def unreachable(model_id: str, revision: str | None) -> list[str]:
        raise ConnectionError("hub unreachable")

    registry, factory = make_registry(lister=unreachable)
    registry.get_or_load("org/cached", pooling=Pooling.MEAN)
    assert factory.model_loads == ["org/cached"]


# --------------------------------------------------------------------------- #
# Eviction
# --------------------------------------------------------------------------- #


def test_idle_model_is_evicted_and_its_backends_torn_down() -> None:
    registry, factory = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN, keep_alive=0.01)
    assert len(registry.list_loaded()) == 1

    time.sleep(0.02)
    assert registry._reap_once() == ["org/model"]

    assert registry.list_loaded() == []
    assert all(backend.closed for backend in factory.created)


def test_model_within_its_keep_alive_survives() -> None:
    registry, _ = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN, keep_alive=60.0)
    assert registry._reap_once() == []
    assert [status.model_id for status in registry.list_loaded()] == ["org/model"]


def test_keep_alive_is_per_entry_not_global() -> None:
    registry, _ = make_registry()
    registry.get_or_load("org/ephemeral", pooling=Pooling.MEAN, keep_alive=0)
    registry.get_or_load("org/pinned", pooling=Pooling.MEAN, keep_alive=60.0)
    assert registry._reap_once() == ["org/ephemeral"]
    assert [status.model_id for status in registry.list_loaded()] == ["org/pinned"]


def test_keep_alive_zero_evicts_on_the_next_pass_without_waiting() -> None:
    registry, factory = make_registry()
    registry.get_or_load("org/once", pooling=Pooling.MEAN, keep_alive=0)
    # No sleeping at all: a purely time-based rule would keep this alive.
    assert registry._reap_once() == ["org/once"]
    assert registry.list_loaded() == []
    assert all(backend.closed for backend in factory.created)


def test_keep_alive_zero_evicts_when_the_triggering_request_completes() -> None:
    # Not "on the next reaper tick": with the default 30s interval that
    # would leave a keep_alive=0 model on the card for half a minute.
    registry, factory = make_registry()
    with registry.acquire("org/once", pooling=Pooling.MEAN, keep_alive=0):
        for _ in range(5):
            assert registry._reap_once() == []  # held, so never collected
        assert registry.list_loaded()[0].ref_count == 1
    assert registry.list_loaded() == [], "eviction must not wait for the reaper"
    assert all(backend.closed for backend in factory.created)


def test_keep_alive_zero_survives_until_the_last_holder_releases() -> None:
    registry, _ = make_registry()
    with registry.acquire("org/once", pooling=Pooling.MEAN, keep_alive=0):
        with registry.acquire("org/once"):
            assert registry.list_loaded()[0].ref_count == 2
        # Inner block done, but the outer one still holds it.
        assert [status.model_id for status in registry.list_loaded()] == ["org/once"]
    assert registry.list_loaded() == []


def test_referenced_model_is_never_evicted_however_long_it_idles() -> None:
    registry, factory = make_registry()
    released = threading.Event()
    holding = threading.Event()

    keep_alive = 0.05

    def holder() -> None:
        with registry.acquire("org/held", pooling=Pooling.MEAN, keep_alive=keep_alive):
            holding.set()
            released.wait(5.0)

    thread = threading.Thread(target=holder)
    thread.start()
    assert holding.wait(5.0)

    time.sleep(keep_alive * 2)  # well past keep_alive, and still held
    for _ in range(20):
        assert registry._reap_once() == []
    assert [status.model_id for status in registry.list_loaded()] == ["org/held"]
    assert registry.list_loaded()[0].ref_count == 1
    assert not any(backend.closed for backend in factory.created)

    released.set()
    thread.join(5.0)
    # Release restarts the idle clock: the TTL measures time since last USE,
    # not time since load, so it is not instantly collectable here.
    assert registry._reap_once() == []
    time.sleep(keep_alive * 2)
    assert registry._reap_once() == ["org/held"]


def test_reference_is_released_when_the_block_raises() -> None:
    registry, _ = make_registry()
    with pytest.raises(ZeroDivisionError), registry.acquire("org/model", pooling=Pooling.MEAN):
        raise ZeroDivisionError
    assert registry.list_loaded()[0].ref_count == 0


def test_engine_from_an_evicted_model_refuses_to_embed() -> None:
    registry, _ = make_registry()
    engine = registry.get_or_load("org/model", pooling=Pooling.MEAN, keep_alive=0)
    registry._reap_once()
    with pytest.raises(RuntimeError, match="closed"):
        engine.embed(["anything"])


def test_reload_after_eviction_loads_again() -> None:
    registry, factory = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN, keep_alive=0)
    registry._reap_once()
    registry.get_or_load("org/model", pooling=Pooling.MEAN, keep_alive=0)
    assert factory.model_loads == ["org/model", "org/model"]


def test_reaper_thread_start_and_stop() -> None:
    registry, factory = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN, keep_alive=0)
    registry.start()
    registry.start()  # idempotent
    deadline = time.monotonic() + 5.0
    while registry.list_loaded() and time.monotonic() < deadline:
        time.sleep(0.01)
    registry.stop()
    registry.stop()  # idempotent
    assert registry.list_loaded() == []
    assert all(backend.closed for backend in factory.created)


def test_evict_all_ignores_references_and_idle_time() -> None:
    registry, factory = make_registry()
    registry.get_or_load("a/model", pooling=Pooling.MEAN, keep_alive=3600)
    registry.get_or_load("b/model", pooling=Pooling.MEAN, keep_alive=3600)
    assert sorted(registry.evict_all()) == ["a/model", "b/model"]
    assert registry.list_loaded() == []
    assert all(backend.closed for backend in factory.created)


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #


def test_list_loaded_tracks_a_full_lifecycle() -> None:
    registry, _ = make_registry()
    assert registry.list_loaded() == []

    registry.get_or_load("org/model", pooling=Pooling.CLS, dtype=Dtype.FLOAT16, keep_alive=0.01)
    loaded = registry.list_loaded()
    assert len(loaded) == 1
    status = loaded[0]
    assert status.model_id == "org/model"
    assert status.pooling is Pooling.CLS
    assert status.dtype is Dtype.FLOAT16
    assert status.device_indices == (0, 1)
    assert status.ref_count == 0
    assert status.idle_s >= 0.0

    with registry.acquire("org/model") as engine:
        held = registry.list_loaded()[0]
        assert held.ref_count == 1
        assert held.idle_s == 0.0, "a model in use is not idle, whatever the clock says"
        engine.embed(["x"])

    assert registry.list_loaded()[0].ref_count == 0

    time.sleep(0.02)
    registry._reap_once()
    assert registry.list_loaded() == []


def test_list_loaded_is_sorted_and_reports_every_model() -> None:
    registry, _ = make_registry()
    for model_id in ("z/model", "a/model", "m/model"):
        registry.get_or_load(model_id, pooling=Pooling.MEAN)
    assert [status.model_id for status in registry.list_loaded()] == [
        "a/model",
        "m/model",
        "z/model",
    ]


def test_model_status_is_frozen() -> None:
    registry, _ = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN)
    status = registry.list_loaded()[0]
    with pytest.raises(AttributeError):
        status.ref_count = 99  # type: ignore[misc]


def test_default_keep_alive_applies_when_unspecified() -> None:
    registry, _ = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN)
    assert DEFAULT_KEEP_ALIVE_S > 0
    assert registry._reap_once() == []


def test_every_registry_error_shares_one_base() -> None:
    # Task 14 maps these to status codes by type, so the hierarchy is part
    # of the contract, not an implementation detail.
    for exc_type in (
        PoolingRequiredError,
        PoolingConflictError,
        ModelPlacementError,
        UnsupportedWeightFormatError,
    ):
        assert issubclass(exc_type, RegistryError)
    assert issubclass(RegistryError, Exception)


def test_pooling_resolution_and_conflict_are_logged_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Both are WARNING by requirement: the resolution fixes how every vector
    # from this model is produced, and the conflict means two callers
    # disagree about it. Neither should need DEBUG logging to be seen.
    registry, _ = make_registry()
    with caplog.at_level(logging.WARNING, logger="embedx.registry"):
        registry.get_or_load("org/model", pooling=Pooling.MEAN)
    resolution = [record for record in caplog.records if "resolved to mean" in record.message]
    assert len(resolution) == 1
    assert resolution[0].levelno == logging.WARNING
    assert "org/model" in resolution[0].getMessage()

    caplog.clear()
    with (
        caplog.at_level(logging.WARNING, logger="embedx.registry"),
        pytest.raises(PoolingConflictError),
    ):
        registry.get_or_load("org/model", pooling=Pooling.CLS)
    conflict = [record for record in caplog.records if "conflict" in record.message]
    assert len(conflict) == 1
    assert conflict[0].levelno == logging.WARNING
    assert "org/model" in conflict[0].getMessage()


def test_status_reports_a_wall_clock_last_used_timestamp() -> None:
    registry, _ = make_registry()
    before = time.time()
    registry.get_or_load("org/model", pooling=Pooling.MEAN)
    after = time.time()

    status = registry.list_loaded()[0]
    assert before <= status.last_used_epoch_s <= after

    time.sleep(0.01)
    with registry.acquire("org/model"):
        pass
    refreshed = registry.list_loaded()[0]
    assert refreshed.last_used_epoch_s > status.last_used_epoch_s
