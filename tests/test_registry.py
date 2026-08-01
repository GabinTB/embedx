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
from embedx.config import Dtype, Pooling, Settings
from embedx.gpu.discovery import DeviceInfo
from embedx.registry import (
    DEFAULT_KEEP_ALIVE_S,
    ModelCapacityError,
    ModelPlacementError,
    ModelRegistry,
    PoolingConflictError,
    PoolingRequiredError,
    RegistryError,
    UnsupportedWeightFormatError,
    WeightFormatUnverifiableError,
    _default_weight_files,
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
        self.truncated_count = 0
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


def test_an_unlistable_repo_is_refused_rather_than_loaded() -> None:
    # Fails CLOSED. A check that silently does not run when the listing
    # fails is not a check, and that is precisely when it matters.
    def unreachable(model_id: str, revision: str | None) -> list[str]:
        raise ConnectionError("hub unreachable: connection timed out")

    registry, factory = make_registry(lister=unreachable)
    with pytest.raises(WeightFormatUnverifiableError) as excinfo:
        registry.get_or_load("org/cached", pooling=Pooling.MEAN)

    assert factory.loads == 0, "nothing may be constructed on an unverified checkpoint"
    assert registry.list_loaded() == []
    message = str(excinfo.value)
    assert "org/cached" in message
    assert "ConnectionError" in message and "connection timed out" in message
    assert excinfo.value.__cause__ is excinfo.value.cause
    assert isinstance(excinfo.value.cause, ConnectionError)


def test_unverifiable_is_a_distinct_type_from_pickle_weights() -> None:
    # Task 14 maps "ships pickle" (permanent, do not retry) and "could not
    # check" (transient, do retry) to different status codes, which it can
    # only do if neither type catches the other.
    assert issubclass(WeightFormatUnverifiableError, RegistryError)
    assert not issubclass(WeightFormatUnverifiableError, UnsupportedWeightFormatError)
    assert not issubclass(UnsupportedWeightFormatError, WeightFormatUnverifiableError)


def test_a_local_directory_is_listed_without_any_network_call(tmp_path: Any) -> None:
    # The real lister, not a stub: a local path must never reach the hub,
    # so failing closed cannot affect it. Any hub call here would raise.
    checkpoint = tmp_path / "my-model"
    (checkpoint / "nested").mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"\x00")
    (checkpoint / "nested" / "extra.json").write_text("{}")

    files = _default_weight_files(str(checkpoint), None)
    assert sorted(files) == ["config.json", "model.safetensors", "nested/extra.json"]

    registry, factory = make_registry(lister=_default_weight_files)
    registry.get_or_load(str(checkpoint), pooling=Pooling.MEAN)
    assert factory.model_loads == [str(checkpoint)]


def test_a_cached_model_falls_back_to_the_snapshot_when_the_hub_is_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # list_repo_files is a pure API call and never consults the cache, so a
    # fully downloaded model raises there on an offline host. Without this
    # fallback, failing closed would refuse every cached model on an
    # air-gapped box — including ones whose safetensors are on disk.
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00")

    def offline(*args: Any, **kwargs: Any) -> Any:
        raise OSError("offline mode is enabled")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", offline)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snapshot))

    assert sorted(_default_weight_files("org/cached", None)) == [
        "config.json",
        "model.safetensors",
    ]

    registry, factory = make_registry(lister=_default_weight_files)
    registry.get_or_load("org/cached", pooling=Pooling.MEAN)
    assert factory.model_loads == ["org/cached"]


def test_nothing_cached_and_hub_down_surfaces_the_hub_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    def offline(*args: Any, **kwargs: Any) -> Any:
        raise OSError("offline mode is enabled")

    def not_cached(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("no local snapshot")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", offline)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)

    registry, factory = make_registry(lister=_default_weight_files)
    with pytest.raises(WeightFormatUnverifiableError) as excinfo:
        registry.get_or_load("org/missing", pooling=Pooling.MEAN)
    # The hub failure is the useful cause, not the cache miss that followed.
    assert "offline mode is enabled" in str(excinfo.value)
    assert factory.loads == 0


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


def test_status_sums_truncation_across_every_device_of_a_model() -> None:
    registry, factory = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN)
    assert registry.list_loaded()[0].truncated_count == 0

    for backend, truncated in zip(factory.created, (11, 2), strict=True):
        backend.truncated_count = truncated
    # One total for the model, not one number per card.
    assert registry.list_loaded()[0].truncated_count == 13


def test_status_truncation_is_zero_for_a_backend_without_the_counter() -> None:
    registry, factory = make_registry()
    registry.get_or_load("org/model", pooling=Pooling.MEAN)
    for backend in factory.created:
        del backend.truncated_count
    assert registry.list_loaded()[0].truncated_count == 0


# --------------------------------------------------------------------------- #
# max_loaded_models
# --------------------------------------------------------------------------- #


def capped_registry(cap: int) -> tuple[ModelRegistry, StubFactory]:
    factory = StubFactory()
    registry = ModelRegistry(
        make_devices(2),
        reaper_interval_s=0.01,
        settings=Settings(max_loaded_models=cap),
        backend_factory=factory,
        weight_file_lister=safetensors_listing,
    )
    return registry, factory


def test_no_cap_by_default() -> None:
    registry, _ = make_registry()
    for index in range(5):
        registry.get_or_load(f"org/model-{index}", pooling=Pooling.MEAN)
    assert len(registry.list_loaded()) == 5


def test_loading_past_the_cap_evicts_the_least_recently_used() -> None:
    registry, factory = capped_registry(2)
    registry.get_or_load("org/first", pooling=Pooling.MEAN)
    time.sleep(0.01)
    registry.get_or_load("org/second", pooling=Pooling.MEAN)
    assert sorted(s.model_id for s in registry.list_loaded()) == ["org/first", "org/second"]

    # Touch the older one so the YOUNGER becomes least-recently-used: the
    # rule is last use, not load order.
    time.sleep(0.01)
    registry.get_or_load("org/first")

    registry.get_or_load("org/third", pooling=Pooling.MEAN)
    assert sorted(s.model_id for s in registry.list_loaded()) == ["org/first", "org/third"]
    # Evicted for real, not just dropped from the map.
    evicted = [b for b in factory.created if b.model_id == "org/second"]
    assert evicted and all(b.closed for b in evicted)


def test_the_cap_never_evicts_a_model_that_is_in_use() -> None:
    registry, _ = capped_registry(1)
    with registry.acquire("org/busy", pooling=Pooling.MEAN):
        with pytest.raises(ModelCapacityError) as excinfo:
            registry.get_or_load("org/wants-in", pooling=Pooling.MEAN)
        assert excinfo.value.resident == ["org/busy"]
        assert excinfo.value.cap == 1
        assert "org/wants-in" in str(excinfo.value)
        # The in-flight model is untouched.
        assert [s.model_id for s in registry.list_loaded()] == ["org/busy"]

    # Released: the same load now succeeds by evicting it.
    registry.get_or_load("org/wants-in", pooling=Pooling.MEAN)
    assert [s.model_id for s in registry.list_loaded()] == ["org/wants-in"]


def test_the_cap_evicts_only_unreferenced_models() -> None:
    registry, _ = capped_registry(2)
    registry.get_or_load("org/idle", pooling=Pooling.MEAN)
    time.sleep(0.01)
    with registry.acquire("org/busy", pooling=Pooling.MEAN):
        # org/busy is the more recently used, but org/idle is the only
        # candidate anyway: being unreferenced comes first.
        registry.get_or_load("org/new", pooling=Pooling.MEAN)
    assert sorted(s.model_id for s in registry.list_loaded()) == ["org/busy", "org/new"]


def test_default_keep_alive_comes_from_settings() -> None:
    factory = StubFactory()
    registry = ModelRegistry(
        make_devices(1),
        settings=Settings(default_keep_alive_s=0.01),
        backend_factory=factory,
        weight_file_lister=safetensors_listing,
    )
    registry.get_or_load("org/model", pooling=Pooling.MEAN)  # no keep_alive given
    assert registry._reap_once() == []
    time.sleep(0.02)
    assert registry._reap_once() == ["org/model"]


def test_engine_batching_budgets_come_from_the_given_settings() -> None:
    factory = StubFactory()
    registry = ModelRegistry(
        make_devices(1),
        settings=Settings(max_batch_tokens=4096),
        backend_factory=factory,
        weight_file_lister=safetensors_listing,
    )
    engine = registry.get_or_load("org/model", pooling=Pooling.MEAN)
    assert [budget for _, budget in engine.devices_with_budgets] == [4096]


# --------------------------------------------------------------------------- #
# max_concurrent_loads
# --------------------------------------------------------------------------- #


class LoadProbe:
    """Records how many cold loads overlap, and the high-water mark."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        with self._lock:
            self.current -= 1


class ProbedFactory(StubFactory):
    """Backend factory that reports overlap and can be held open."""

    def __init__(self, probe: LoadProbe, delay_s: float = 0.1, gate: Any = None) -> None:
        super().__init__()
        self.probe = probe
        self.load_delay_s = delay_s
        self.gate = gate

    def __call__(self, **kwargs: Any) -> StubBackend:
        self.probe.enter()
        try:
            if self.gate is not None:
                self.gate.wait(5.0)
            time.sleep(self.load_delay_s)
            return super().__call__(**kwargs)
        finally:
            self.probe.exit()


def concurrent_loads(registry: ModelRegistry, model_ids: list[str]) -> list[BaseException | None]:
    errors: list[BaseException | None] = [None] * len(model_ids)

    def worker(index: int, model_id: str) -> None:
        try:
            with registry.acquire(model_id, pooling=Pooling.MEAN):
                pass
        except BaseException as exc:  # surfaced by the caller's assert
            errors[index] = exc

    threads = [
        threading.Thread(target=worker, args=(index, model_id))
        for index, model_id in enumerate(model_ids)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    return errors


def test_cold_loads_are_capped_independently_of_requests() -> None:
    probe = LoadProbe()
    factory = ProbedFactory(probe, delay_s=0.1)
    registry = ModelRegistry(
        make_devices(1),
        settings=Settings(max_concurrent_loads=2, max_concurrent_requests=8),
        backend_factory=factory,
        weight_file_lister=safetensors_listing,
    )
    errors = concurrent_loads(registry, [f"org/model-{index}" for index in range(6)])

    assert errors == [None] * 6
    assert len(registry.list_loaded()) == 6, "every model must still load, just not at once"
    assert probe.peak <= 2, f"load cap breached: {probe.peak} loads ran at once"
    assert probe.peak > 1, "loads never overlapped; the test proves nothing"


def test_a_cap_of_one_serialises_loads_completely() -> None:
    probe = LoadProbe()
    factory = ProbedFactory(probe, delay_s=0.05)
    registry = ModelRegistry(
        make_devices(1),
        settings=Settings(max_concurrent_loads=1, max_concurrent_requests=4),
        backend_factory=factory,
        weight_file_lister=safetensors_listing,
    )
    assert concurrent_loads(registry, [f"org/model-{i}" for i in range(4)]) == [None] * 4
    assert probe.peak == 1


def test_a_warm_acquire_never_consumes_a_load_slot() -> None:
    """The test that proves the two caps are actually separate.

    Every cold-load slot is held open by a load that will not finish until
    released. If a warm acquire touched the load semaphore, this would block
    until that gate opens, and the load cap would silently have become a
    second request cap.
    """
    probe = LoadProbe()
    gate = threading.Event()
    factory = ProbedFactory(probe, delay_s=0.0, gate=gate)
    registry = ModelRegistry(
        make_devices(1),
        settings=Settings(max_concurrent_loads=1, max_concurrent_requests=4),
        backend_factory=factory,
        weight_file_lister=safetensors_listing,
    )

    # Load one model to completion, so it is warm.
    gate.set()
    registry.get_or_load("org/warm", pooling=Pooling.MEAN)
    assert registry.in_flight_loads == 0

    # Now occupy the only load slot with a cold load that hangs.
    gate.clear()
    blocked = threading.Thread(
        target=lambda: registry.get_or_load("org/cold", pooling=Pooling.MEAN)
    )
    blocked.start()
    try:
        deadline = time.monotonic() + 5.0
        while registry.in_flight_loads < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert registry.in_flight_loads == 1, "the cold load never took its slot"

        # The load cap is now fully occupied. A warm acquire must not care.
        done = threading.Event()

        def warm() -> None:
            with registry.acquire("org/warm") as engine:
                engine.embed(["x"])
            done.set()

        threading.Thread(target=warm).start()
        assert done.wait(3.0), "a warm acquire blocked on the cold-load semaphore"
    finally:
        gate.set()
        blocked.join(10)


def test_in_flight_loads_returns_to_zero_after_a_failed_load() -> None:
    # A slot leaked on the error path would shrink the cap permanently.
    probe = LoadProbe()

    class FailingFactory(ProbedFactory):
        def __call__(self, **kwargs: Any) -> StubBackend:
            raise ValueError("checkpoint is broken")

    registry = ModelRegistry(
        make_devices(1),
        settings=Settings(max_concurrent_loads=1, max_concurrent_requests=2),
        backend_factory=FailingFactory(probe),
        weight_file_lister=safetensors_listing,
    )
    for _ in range(3):
        with pytest.raises(ValueError, match="broken"):
            registry.get_or_load("org/bad", pooling=Pooling.MEAN)
        assert registry.in_flight_loads == 0

    # The slot survived: a real load still works afterwards.
    registry._backend_factory = StubFactory()  # type: ignore[assignment]
    registry.get_or_load("org/good", pooling=Pooling.MEAN)
    assert [s.model_id for s in registry.list_loaded()] == ["org/good"]


def test_load_caps_are_reported_for_info() -> None:
    registry, _ = make_registry()
    assert registry.in_flight_loads == 0
    assert registry.max_concurrent_loads == Settings().max_concurrent_loads
