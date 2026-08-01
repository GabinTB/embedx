"""Device discovery and static ranking, over the vendor seam.

Importable without torch: this module no longer touches a tensor library
at all. Enumeration, device properties and the architecture score come
from an `Accelerator` (see `embedx.gpu.vendor`), so everything here works
on plain-typed `DeviceInfo` values and stays CPU-testable.

`DeviceInfo` and `ARCH_FACTOR` are re-exported from `vendor` rather than
defined here — the dataclass is the seam's own data type, and the score
table is CUDA-specific, so both belong beside the implementation that
reads them. The names stay importable from this module because callers
across the codebase use them and this commit changes no behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from embedx.gpu.vendor import ARCH_FACTOR, Accelerator, DeviceInfo, get_accelerator

__all__ = ["ARCH_FACTOR", "DeviceInfo", "discover_devices", "rank_devices"]


def discover_devices(
    devices: list[int] | None = None, accelerator: Accelerator | None = None
) -> list[DeviceInfo]:
    """Enumerate visible devices; `[]` when no accelerator runtime is usable.

    `devices` is the config field of the same name: when set, the result is
    filtered (and ordered) to those indices, and a requested index that is
    not present raises rather than being silently dropped.

    `accelerator` defaults to the process's real one; tests inject a fake.
    """
    accel = accelerator if accelerator is not None else get_accelerator()
    if not accel.is_available():
        return []
    infos = [accel.device_info(index) for index in range(accel.device_count())]
    if devices is None:
        return infos
    by_index = {info.index: info for info in infos}
    missing = sorted(set(devices) - set(by_index))
    if missing:
        raise ValueError(
            f"requested device indices {missing} not present; visible devices: {sorted(by_index)}"
        )
    return [by_index[index] for index in devices]


def rank_devices(
    infos: Iterable[DeviceInfo],
    weights_override: Mapping[int, float],
    accelerator: Accelerator | None = None,
) -> list[DeviceInfo]:
    """Rank devices fastest-first by static score, then config overrides.

    The score is the accelerator's `arch_score` — on CUDA,
    `multi_processor_count * ARCH_FACTOR[capability major]`; weight is the
    score normalised so the best device is 1.0. An entry in
    `weights_override` (`Settings.device_weights`) replaces the computed
    weight outright and therefore reorders the ranking — that is the point:
    a card on a narrow PCIe link ranks far below what its SM count
    suggests, and the user must be able to say so. Ties break by index
    ascending for determinism. Returns new instances; input is not mutated.
    """
    accel = accelerator if accelerator is not None else get_accelerator()
    scored = [replace(info, score=accel.arch_score(info)) for info in infos]
    if not scored:
        return []
    top = max(info.score for info in scored)
    ranked = [
        replace(
            info,
            weight=weights_override.get(info.index, info.score / top if top > 0 else 1.0),
        )
        for info in scored
    ]
    ranked.sort(key=lambda info: (-info.weight, info.index))
    return ranked
