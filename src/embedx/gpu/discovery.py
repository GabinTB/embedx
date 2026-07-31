"""GPU discovery and static ranking.

Importable without torch: the import happens lazily inside
`discover_devices`, and everything else works on plain-typed `DeviceInfo`
values, so ranking and budget logic stays CPU-testable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import ModuleType

# Rough per-SM throughput multiplier by compute capability major version.
# torch exposes neither clock rate nor memory bandwidth, so this is a coarse
# static proxy — deliberately correctable by config, which is why
# `device_weights` exists at all. Values are relative, not benchmarks.
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
    """One CUDA device, described with plain types only — no torch types."""

    index: int
    name: str
    total_memory_bytes: int
    multi_processor_count: int
    capability: tuple[int, int]
    score: float = 0.0
    weight: float = 1.0


def _import_torch() -> ModuleType | None:
    """Lazy torch import; None when absent. Monkeypatchable in tests."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def discover_devices(devices: list[int] | None = None) -> list[DeviceInfo]:
    """Enumerate visible CUDA devices; `[]` without torch or without CUDA.

    `devices` is the config field of the same name: when set, the result is
    filtered (and ordered) to those indices, and a requested index that is
    not present raises rather than being silently dropped.
    """
    torch = _import_torch()
    if torch is None or not torch.cuda.is_available():
        return []
    infos = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        infos.append(
            DeviceInfo(
                index=index,
                name=str(props.name),
                total_memory_bytes=int(props.total_memory),
                multi_processor_count=int(props.multi_processor_count),
                capability=(int(props.major), int(props.minor)),
            )
        )
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
    infos: Iterable[DeviceInfo], weights_override: Mapping[int, float]
) -> list[DeviceInfo]:
    """Rank devices fastest-first by static score, then config overrides.

    `score = multi_processor_count * ARCH_FACTOR[capability major]`; weight
    is the score normalised so the best device is 1.0. An entry in
    `weights_override` (`Settings.device_weights`) replaces the computed
    weight outright and therefore reorders the ranking — that is the point:
    a card on a narrow PCIe link ranks far below what its SM count
    suggests, and the user must be able to say so. Ties break by index
    ascending for determinism. Returns new instances; input is not mutated.
    """
    scored = [
        replace(info, score=info.multi_processor_count * _arch_factor(info.capability[0]))
        for info in infos
    ]
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
