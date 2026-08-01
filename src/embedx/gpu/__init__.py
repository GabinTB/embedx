"""GPU discovery, ranking, and budgets. Importable without torch."""

from embedx.gpu.budgets import device_budgets
from embedx.gpu.discovery import discover_devices, rank_devices
from embedx.gpu.vendor import ARCH_FACTOR, Accelerator, DeviceInfo, get_accelerator

__all__ = [
    "ARCH_FACTOR",
    "Accelerator",
    "DeviceInfo",
    "device_budgets",
    "discover_devices",
    "get_accelerator",
    "rank_devices",
]
