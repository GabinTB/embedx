"""Per-device token budgets, scaled by memory.

The token budget bounds activation memory, so it scales with each card's
memory — not its speed: a slow card with lots of RAM can still take big
batches; a fast card with little RAM cannot.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from embedx.gpu.discovery import DeviceInfo


def device_budgets(
    ranked: Iterable[DeviceInfo],
    default_tokens: int,
    overrides: Mapping[int, int],
    min_tokens: int = 512,
) -> dict[int, int]:
    """Token budget per device index.

    `budget_i = max(min_tokens, int(default_tokens * mem_i / max_mem))`
    where `max_mem` is the largest memory among `ranked`, so the biggest
    card gets exactly `default_tokens` (`Settings.max_batch_tokens`). An
    entry in `overrides` (`Settings.device_batch_tokens`) is used verbatim
    with no scaling.
    """
    if default_tokens <= 0:
        raise ValueError(f"default_tokens must be > 0, got {default_tokens}")
    if min_tokens <= 0:
        raise ValueError(f"min_tokens must be > 0, got {min_tokens}")
    devices = list(ranked)
    if not devices:
        return {}
    max_mem = max(device.total_memory_bytes for device in devices)
    budgets: dict[int, int] = {}
    for device in devices:
        if device.index in overrides:
            budgets[device.index] = overrides[device.index]
            continue
        scale = device.total_memory_bytes / max_mem if max_mem > 0 else 1.0
        budgets[device.index] = max(min_tokens, int(default_tokens * scale))
    return budgets
