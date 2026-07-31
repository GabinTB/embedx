"""GPU discovery, ranking, and budgets. Importable without torch."""

from embedx.gpu.budgets import device_budgets
from embedx.gpu.discovery import ARCH_FACTOR, DeviceInfo, discover_devices, rank_devices

__all__ = ["ARCH_FACTOR", "DeviceInfo", "device_budgets", "discover_devices", "rank_devices"]
