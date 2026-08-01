"""The vendor seam: every CUDA-specific call behind one Protocol.

embedx is CUDA-only and stays CUDA-only here. The point of this module is
that adding ROCm or XPU later becomes *a new class implementing
`Accelerator`* rather than an edit spanning discovery, the backend, the
registry and the CLI. Nothing outside this file may touch `torch.cuda`;
the ast guard in `tests/test_gpu.py` enforces that, and without the guard
the seam would rot within a couple of commits.

No torch at import time, as everywhere else in the core: `_import_torch`
is called inside methods, so `import embedx.gpu.vendor` works on a host
with no GPU stack at all.

Why `DeviceInfo.capability` survives as a compute-capability pair
---------------------------------------------------------------
The task allowed replacing it with a vendor-neutral field. It stays, for
two reasons.

First, it is load-bearing in the existing suite: `capability=(8, 0)` is a
constructor keyword in six test modules, and this commit is a pure
refactor whose whole claim is that nothing observable changed. Renaming
the field would mean editing those tests, which is precisely the signal
the task says to treat as a leak.

Second, and more importantly, the decoupling already happened without a
rename. `capability` now has exactly two readers, `arch_score` and
`supports_bfloat16`/`supports_float16`, and both live *inside* an
Accelerator implementation. No core module reads it any more. The field
is therefore opaque vendor-private state that happens to be typed
`tuple[int, int]`: a ROCm implementation is free to populate it with a
gfx major/minor and interpret it in its own methods, and nothing outside
this file would notice. Renaming it would be cosmetic — the coupling that
mattered was the `capability.major >= 8` comparison inlined in the
backend, and that is gone.

Apple MPS is deliberately out of scope
--------------------------------------
MPS is a fourth backend that does *not* fit this Protocol cleanly: it has
unified memory, so there is no discrete VRAM budget to scale token
budgets against, no per-device enumeration worth the name, and
`empty_cache` means something different. Forcing it in would mean
`total_memory_bytes` lying to `device_budgets`. It needs its own seam, or
a different placement story entirely — not an implementation of this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

# Rough per-SM throughput multiplier by compute capability major version.
# torch exposes neither clock rate nor memory bandwidth, so this is a coarse
# static proxy — deliberately correctable by config, which is why
# `device_weights` exists at all. Values are relative, not benchmarks.
#
# CUDA-specific by construction: compute capability has no meaning on ROCm
# or XPU, which is why this table lives beside `CudaAccelerator` rather
# than in `discovery`. Another vendor brings its own table, with the honest
# admission that it is uncalibrated until someone measures on that hardware.
ARCH_FACTOR: dict[int, float] = {
    5: 0.5,  # Maxwell
    6: 0.7,  # Pascal
    7: 1.0,  # Volta / Turing
    8: 1.6,  # Ampere / Ada
    9: 2.6,  # Hopper
    10: 3.2,  # Blackwell
    12: 3.2,  # Blackwell (consumer)
}


def _arch_factor(major: int) -> float:
    """Factor for `major`, falling back to the nearest known lower major.

    Never silently 1.0: an unknown newer architecture inherits the best
    older estimate; a major below everything known gets the lowest entry.
    """
    if major in ARCH_FACTOR:
        return ARCH_FACTOR[major]
    lower = [known for known in ARCH_FACTOR if known < major]
    if lower:
        return ARCH_FACTOR[max(lower)]
    return ARCH_FACTOR[min(ARCH_FACTOR)]


@dataclass(frozen=True)
class DeviceInfo:
    """One accelerator device, described with plain types only.

    No torch types, so ranking and budget logic stays CPU-testable. See the
    module docstring on why `capability` keeps its CUDA shape.
    """

    index: int
    name: str
    total_memory_bytes: int
    multi_processor_count: int
    capability: tuple[int, int]
    score: float = 0.0
    weight: float = 1.0


class Accelerator(Protocol):
    """Everything embedx needs to know about a GPU vendor's runtime."""

    #: Vendor tag for logs, error messages and `/info` — "cuda" today.
    name: str

    def is_available(self) -> bool:
        """True when the runtime is importable and sees at least one device."""
        ...

    def device_count(self) -> int: ...

    def device_info(self, index: int) -> DeviceInfo:
        """Describe one device. Only valid for `index < device_count()`."""
        ...

    def device_string(self, index: int) -> str:
        """What the tensor framework calls this device — `"cuda:0"` today."""
        ...

    def oom_error_types(self) -> tuple[type[BaseException], ...]:
        """Exception types meaning "this model did not fit on this device".

        A tuple, not a single type: vendors raise different things and some
        raise more than one. Empty when the runtime is not importable, which
        is the normal case in CPU CI — see `oom_error_names`.
        """
        ...

    def oom_error_names(self) -> tuple[str, ...]:
        """Type *names* of the same errors, for when the runtime is absent.

        Placement retry is core logic tested on CPU, where the real
        exception classes cannot be imported to `isinstance` against. Each
        vendor declares the names it raises so the fallback stays that
        vendor's business rather than a hardcoded CUDA string in the
        registry. A vendor whose errors are always importable returns `()`.
        """
        ...

    def empty_cache(self, index: int) -> None:
        """Return one device's cached blocks to the driver; never raises."""
        ...

    def supports_bfloat16(self, info: DeviceInfo) -> bool: ...

    def supports_float16(self, info: DeviceInfo) -> bool: ...

    def arch_score(self, info: DeviceInfo) -> float:
        """Static per-device throughput proxy; higher is faster."""
        ...


def _import_torch() -> ModuleType | None:
    """Lazy torch import; None when absent. Monkeypatchable in tests."""
    try:
        import torch
    except ImportError:
        return None
    return torch


class CudaAccelerator:
    """The one real `Accelerator`: NVIDIA CUDA through torch.

    Every method imports torch lazily, exactly as `discovery` did before
    the seam existed, so constructing one costs nothing and works on a
    torch-free host.
    """

    name: str = "cuda"

    def is_available(self) -> bool:
        torch = _import_torch()
        return torch is not None and bool(torch.cuda.is_available())

    def device_count(self) -> int:
        torch = _import_torch()
        if torch is None:
            return 0
        return int(torch.cuda.device_count())

    def device_info(self, index: int) -> DeviceInfo:
        torch = _import_torch()
        if torch is None:  # pragma: no cover - guarded by is_available()
            raise RuntimeError("torch is not installed; no CUDA device to describe")
        props = torch.cuda.get_device_properties(index)
        return DeviceInfo(
            index=index,
            name=str(props.name),
            total_memory_bytes=int(props.total_memory),
            multi_processor_count=int(props.multi_processor_count),
            capability=(int(props.major), int(props.minor)),
        )

    def device_string(self, index: int) -> str:
        return f"cuda:{index}"

    def oom_error_types(self) -> tuple[type[BaseException], ...]:
        torch = _import_torch()
        if torch is None:
            return ()
        return (torch.cuda.OutOfMemoryError,)

    def oom_error_names(self) -> tuple[str, ...]:
        return ("OutOfMemoryError",)

    def empty_cache(self, index: int) -> None:
        torch = _import_torch()
        if torch is None or not torch.cuda.is_available():
            return
        with torch.cuda.device(index):
            torch.cuda.empty_cache()

    def supports_bfloat16(self, info: DeviceInfo) -> bool:
        # bfloat16 needs Ampere (compute capability major >= 8).
        return info.capability[0] >= 8

    def supports_float16(self, info: DeviceInfo) -> bool:
        # float16 is solid from Volta/Turing (major >= 7); older computes fp32.
        return info.capability[0] >= 7

    def arch_score(self, info: DeviceInfo) -> float:
        return info.multi_processor_count * _arch_factor(info.capability[0])


# The single place a future vendor gets selected. When ROCm or XPU lands,
# this is where the choice happens — probing each runtime in turn, or
# reading a config field / EMBEDX_ACCELERATOR env var. One implementation
# today, so it stays a constant: a resolver that can only return one thing
# should look like one.
_CUDA = CudaAccelerator()


def get_accelerator() -> Accelerator:
    """The accelerator this process uses."""
    return _CUDA
