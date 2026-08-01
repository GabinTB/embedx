"""Tests for the vendor seam (task 18).

The point of this module is negative: everything here runs with no torch
involved at all, so anything that still reached for a CUDA symbol would
fail on the import rather than quietly working on the GPU host and
breaking on CPU CI.

`FakeAccelerator` is to `Accelerator` what `FakeBackend` is to
`EmbeddingBackend` — the seam's proof of existence, not a convenience.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import embedx.gpu.vendor as vendor
from embedx.config import Dtype, Pooling, Settings
from embedx.gpu import DeviceInfo, device_budgets, discover_devices, rank_devices
from embedx.gpu.vendor import ARCH_FACTOR, Accelerator, CudaAccelerator, get_accelerator
from embedx.registry import ModelPlacementError, ModelRegistry

GIB = 2**30


class FakeVendorOOMError(Exception):
    """A vendor OOM whose name is deliberately NOT torch's.

    `tests/test_registry.py` relies on the name-based fallback for a class
    called exactly `OutOfMemoryError`, which is how CPU CI exercises CUDA
    placement without torch. Naming this one differently is what makes this
    module's placement test meaningful: it can only pass through
    `oom_error_types()`, never through that fallback.
    """


class FakeAccelerator:
    """An `Accelerator` for hardware that does not exist. Never imports torch."""

    name = "fake"

    def __init__(
        self,
        memories: list[int] | None = None,
        scores: list[float] | None = None,
        available: bool = True,
        bf16: bool = True,
        fp16: bool = True,
    ) -> None:
        self._memories = memories if memories is not None else [16 * GIB, 4 * GIB]
        self._scores = scores if scores is not None else [100.0, 40.0]
        self._available = available
        self._bf16 = bf16
        self._fp16 = fp16
        self.emptied: list[int] = []

    def is_available(self) -> bool:
        return self._available and bool(self._memories)

    def device_count(self) -> int:
        return len(self._memories)

    def device_info(self, index: int) -> DeviceInfo:
        return DeviceInfo(
            index=index,
            name=f"Fake Accelerator {index}",
            total_memory_bytes=self._memories[index],
            multi_processor_count=64,
            capability=(0, 0),  # meaningless for this vendor, and unread
        )

    def device_string(self, index: int) -> str:
        return f"fake:{index}"

    def oom_error_types(self) -> tuple[type[BaseException], ...]:
        return (FakeVendorOOMError,)

    def oom_error_names(self) -> tuple[str, ...]:
        return ()

    def empty_cache(self, index: int) -> None:
        self.emptied.append(index)

    def supports_bfloat16(self, info: DeviceInfo) -> bool:
        return self._bf16

    def supports_float16(self, info: DeviceInfo) -> bool:
        return self._fp16

    def arch_score(self, info: DeviceInfo) -> float:
        return self._scores[info.index]


def test_fake_satisfies_the_protocol() -> None:
    accel: Accelerator = FakeAccelerator()
    assert accel.name == "fake"


# --------------------------------------------------------------------------- #
# End to end on a vendor that is not CUDA
# --------------------------------------------------------------------------- #


def test_discovery_ranking_and_budgets_run_on_a_non_cuda_vendor() -> None:
    """The test that proves the vendor surface is genuinely closed.

    Discovery through budgets with no torch anywhere: anything that leaked
    a `torch.cuda` call would raise instead of returning these numbers.
    """
    accel = FakeAccelerator(memories=[16 * GIB, 4 * GIB], scores=[100.0, 40.0])

    infos = discover_devices(accelerator=accel)
    assert [info.index for info in infos] == [0, 1]
    assert infos[0].name == "Fake Accelerator 0"

    ranked = rank_devices(infos, {}, accelerator=accel)
    assert [info.index for info in ranked] == [0, 1]
    assert ranked[0].score == 100.0
    assert ranked[1].weight == pytest.approx(0.4)  # 40 / 100

    assert device_budgets(ranked, 16384, {}) == {0: 16384, 1: 4096}


def test_discovery_filters_and_reports_missing_indices_on_a_fake_vendor() -> None:
    accel = FakeAccelerator(memories=[16 * GIB, 4 * GIB])
    assert [info.index for info in discover_devices([1], accelerator=accel)] == [1]
    with pytest.raises(ValueError, match=r"\[3\]"):
        discover_devices([0, 3], accelerator=accel)


def test_unavailable_vendor_discovers_nothing() -> None:
    assert discover_devices(accelerator=FakeAccelerator(available=False)) == []
    assert discover_devices([0, 1], accelerator=FakeAccelerator(available=False)) == []


def test_ranking_uses_the_accelerators_score_not_a_cuda_table() -> None:
    """A vendor whose ordering contradicts ARCH_FACTOR must still win.

    Both fake devices report `capability=(0, 0)`, which the CUDA table maps
    to its lowest factor. If any scoring were still inlined, the SM counts
    (identical) would decide and the ranking would not invert here.
    """
    accel = FakeAccelerator(memories=[8 * GIB, 8 * GIB], scores=[10.0, 90.0])
    ranked = rank_devices(discover_devices(accelerator=accel), {}, accelerator=accel)
    assert [info.index for info in ranked] == [1, 0]


# --------------------------------------------------------------------------- #
# Placement and OOM through the seam
# --------------------------------------------------------------------------- #


def _registry_with(accel: FakeAccelerator, factory: object) -> ModelRegistry:
    infos = discover_devices(accelerator=accel)
    ranked = rank_devices(infos, {}, accelerator=accel)
    return ModelRegistry(
        ranked,
        settings=Settings(),
        backend_factory=factory,  # type: ignore[arg-type]
        weight_file_lister=lambda model_id, revision: ["model.safetensors"],
        accelerator=accel,
    )


def test_placement_retries_on_the_vendors_own_oom_type() -> None:
    """The fake's OOM is caught, and its name is not torch's.

    If the registry still matched `OutOfMemoryError` by name, or caught
    `torch.cuda.OutOfMemoryError` by type, device 0 failing would propagate
    instead of falling through to device 1.
    """
    accel = FakeAccelerator(memories=[16 * GIB, 16 * GIB])
    built: list[int] = []

    def factory(*, device_index: int, **kwargs: object) -> object:
        if device_index == 0:
            raise FakeVendorOOMError("no room on fake device 0")
        built.append(device_index)
        return SimpleNamespace(embed=lambda texts: [], length_fn=len, close=lambda: None)

    registry = _registry_with(accel, factory)
    with registry.acquire("some/model", pooling=Pooling.MEAN, dtype=Dtype.AUTO):
        pass

    assert built == [1]  # placed on the second device after the first OOMed
    assert accel.emptied == [0]  # and the failed device's cache was released


def test_a_non_oom_failure_still_propagates_on_a_fake_vendor() -> None:
    accel = FakeAccelerator(memories=[16 * GIB, 16 * GIB])

    def factory(**kwargs: object) -> object:
        raise ValueError("bad model id")

    registry = _registry_with(accel, factory)
    with (
        pytest.raises(ValueError, match="bad model id"),
        registry.acquire("some/model", pooling=Pooling.MEAN),
    ):
        pass


def test_oom_on_every_device_reports_placement_failure() -> None:
    accel = FakeAccelerator(memories=[16 * GIB, 16 * GIB])

    def factory(*, device_index: int, **kwargs: object) -> object:
        raise FakeVendorOOMError(f"no room on fake device {device_index}")

    registry = _registry_with(accel, factory)
    with (
        pytest.raises(ModelPlacementError) as excinfo,
        registry.acquire("some/model", pooling=Pooling.MEAN),
    ):
        pass
    assert [index for index, _ in excinfo.value.failures] == [0, 1]
    assert accel.emptied == [0, 1]


def test_eviction_releases_through_the_accelerator() -> None:
    accel = FakeAccelerator(memories=[16 * GIB])

    def factory(**kwargs: object) -> object:
        return SimpleNamespace(embed=lambda texts: [], length_fn=len, close=lambda: None)

    registry = _registry_with(accel, factory)
    # keep_alive=0 evicts the moment the last reference drops.
    with registry.acquire("some/model", pooling=Pooling.MEAN, keep_alive=0):
        pass
    assert accel.emptied == [0]


# --------------------------------------------------------------------------- #
# supports_bfloat16 is consulted, not reimplemented downstream
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("bf16", "fp16", "expected"),
    [(True, True, "bfloat16"), (False, True, "float16"), (False, False, "float32")],
)
def test_auto_dtype_asks_the_accelerator(bf16: bool, fp16: bool, expected: str) -> None:
    """AUTO resolution follows the accelerator, not a capability comparison.

    The fake reports `capability=(0, 0)` throughout, so any surviving
    `capability.major >= 8` test would resolve float32 every time and the
    bfloat16 case could not pass.
    """
    from embedx.backend.hf import _resolve_dtype

    accel = FakeAccelerator(bf16=bf16, fp16=fp16)
    torch_stub = SimpleNamespace(bfloat16="bfloat16", float16="float16", float32="float32")
    resolved = _resolve_dtype(Dtype.AUTO, accel.device_info(0), accel, torch_stub)
    assert resolved == expected


@pytest.mark.parametrize("dtype", [Dtype.FLOAT32, Dtype.FLOAT16, Dtype.BFLOAT16])
def test_explicit_dtype_never_asks_the_accelerator(dtype: Dtype) -> None:
    """An explicit dtype is honoured even when the vendor says no.

    Same rule as before the seam: the user's choice is not second-guessed.
    """
    from embedx.backend.hf import _resolve_dtype

    accel = FakeAccelerator(bf16=False, fp16=False)
    torch_stub = SimpleNamespace(bfloat16="bfloat16", float16="float16", float32="float32")
    assert _resolve_dtype(dtype, accel.device_info(0), accel, torch_stub) == dtype.value


# --------------------------------------------------------------------------- #
# CudaAccelerator reproduces the pre-refactor behaviour exactly
# --------------------------------------------------------------------------- #


def cuda_device(index: int = 0, sms: int = 100, major: int = 8, minor: int = 0) -> DeviceInfo:
    """The same synthetic device the task-06 tests build."""
    return DeviceInfo(
        index=index,
        name=f"Fake GPU {index}",
        total_memory_bytes=16 * GIB,
        multi_processor_count=sms,
        capability=(major, minor),
    )


@pytest.mark.parametrize("major", sorted(ARCH_FACTOR))
def test_arch_score_matches_the_table_verbatim(major: int) -> None:
    accel = CudaAccelerator()
    assert accel.arch_score(cuda_device(sms=100, major=major)) == pytest.approx(
        100 * ARCH_FACTOR[major]
    )


def test_arch_score_unknown_major_falls_back_to_nearest_lower() -> None:
    accel = CudaAccelerator()
    highest = max(ARCH_FACTOR)
    assert accel.arch_score(cuda_device(major=highest + 1)) == pytest.approx(
        accel.arch_score(cuda_device(major=highest))
    )
    # Below every known major: the lowest known factor, never a silent 1.0.
    assert accel.arch_score(cuda_device(major=3)) == pytest.approx(
        accel.arch_score(cuda_device(major=min(ARCH_FACTOR)))
    )


def test_arch_score_scales_linearly_with_sm_count() -> None:
    accel = CudaAccelerator()
    assert accel.arch_score(cuda_device(sms=200)) == pytest.approx(
        2 * accel.arch_score(cuda_device(sms=100))
    )


@pytest.mark.parametrize(
    ("major", "bf16", "fp16"),
    [(6, False, False), (7, False, True), (8, True, True), (9, True, True)],
)
def test_cuda_dtype_support_matches_the_old_inline_rule(major: int, bf16: bool, fp16: bool) -> None:
    """bf16 from Ampere (>=8), fp16 from Volta/Turing (>=7) — unchanged."""
    accel = CudaAccelerator()
    assert accel.supports_bfloat16(cuda_device(major=major)) is bf16
    assert accel.supports_float16(cuda_device(major=major)) is fp16


def test_cuda_device_string() -> None:
    assert CudaAccelerator().device_string(3) == "cuda:3"


def test_get_accelerator_returns_cuda() -> None:
    accel = get_accelerator()
    assert accel.name == "cuda"
    assert isinstance(accel, CudaAccelerator)


def test_cuda_oom_names_cover_the_torchless_case() -> None:
    """CPU CI cannot import the class, so the name is the only handle."""
    assert "OutOfMemoryError" in CudaAccelerator().oom_error_names()


# --------------------------------------------------------------------------- #
# CudaAccelerator against a fake torch (relocated from test_gpu.py, task 06)
# --------------------------------------------------------------------------- #
#
# These moved here with `_import_torch`: they always tested how the CUDA
# device enumeration talks to torch, which is now this module's job rather
# than discovery's. Assertions are unchanged — only the patched module name
# differs.


def _fake_torch(properties: list[SimpleNamespace], available: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: len(properties),
            get_device_properties=lambda index: properties[index],
        )
    )


def _fake_props(name: str, memory: int, sms: int, major: int, minor: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, total_memory=memory, multi_processor_count=sms, major=major, minor=minor
    )


def test_discover_without_torch_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vendor, "_import_torch", lambda: None)
    assert discover_devices() == []
    assert discover_devices([0, 1]) == []


def test_discover_without_cuda_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vendor, "_import_torch", lambda: _fake_torch([], available=False))
    assert discover_devices() == []


def test_discover_against_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    props = [
        _fake_props("Fake A100", 40 * GIB, 108, 8, 0),
        _fake_props("Fake T4", 16 * GIB, 40, 7, 5),
    ]
    monkeypatch.setattr(vendor, "_import_torch", lambda: _fake_torch(props))

    infos = discover_devices()
    assert [info.index for info in infos] == [0, 1]
    assert infos[0] == DeviceInfo(
        index=0,
        name="Fake A100",
        total_memory_bytes=40 * GIB,
        multi_processor_count=108,
        capability=(8, 0),
    )

    filtered = discover_devices(devices=[1])
    assert [info.index for info in filtered] == [1]
    assert filtered[0].name == "Fake T4"

    with pytest.raises(ValueError, match=r"\[3\]"):
        discover_devices(devices=[0, 3])


def test_empty_cache_is_a_noop_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vendor, "_import_torch", lambda: None)
    CudaAccelerator().empty_cache(0)  # must not raise


def test_oom_error_types_is_empty_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vendor, "_import_torch", lambda: None)
    assert CudaAccelerator().oom_error_types() == ()
