"""Core engine: batching, scheduling, and result reassembly. Torch-free."""

from embedx.engine.batching import make_batches, pack_count

__all__ = ["make_batches", "pack_count"]
