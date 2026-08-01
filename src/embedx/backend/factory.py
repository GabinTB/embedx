"""Backend construction helpers, shared by whatever assembles an Engine.

Importable without torch; the HFBackend import happens inside the helpers
below, which are the first point where torch is genuinely needed.

`build_engine`, the eager single-model startup path, is gone: `serve` now
starts with nothing loaded and `ModelRegistry` places each model device by
device on demand. What it needed survives here as separate functions — a
backend constructor, the per-model token-length cache, and the Engine
wiring — because the registry needs exactly those, one device at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from embedx.backend.base import EmbeddingBackend
from embedx.config import Dtype, Pooling, Settings
from embedx.engine.engine import Engine
from embedx.gpu.discovery import DeviceInfo


class BackendFactory(Protocol):
    """How a backend is constructed for one device.

    The seam the registry injects a fake through: keyword-only, so a test
    double can accept exactly the arguments it cares about.
    """

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
    ) -> EmbeddingBackend: ...


def new_length_cache() -> Any:
    """One `TokenLengthCache`, to be shared by every device of one model.

    The engine batches with `backends[0].length_fn`, so a per-backend cache
    would leave every other device cold and re-tokenizing whole batches in
    `_count_truncations`. Same model, same tokenizer — entries are valid on
    every device.
    """
    from embedx.backend.hf import TokenLengthCache  # lazy: hf pulls in torch

    return TokenLengthCache()


def hf_backend_factory(
    *,
    model_id: str,
    device_index: int,
    pooling: Pooling,
    normalize: bool,
    dtype: Dtype,
    max_seq_length: int | None,
    revision: str | None,
    length_cache: Any,
) -> EmbeddingBackend:
    """The real backend constructor; the default `BackendFactory`."""
    from embedx.backend.hf import HFBackend  # lazy: pulls in torch

    return HFBackend(
        model_id=model_id,
        device_index=device_index,
        pooling=pooling,
        normalize=normalize,
        dtype=dtype,
        max_seq_length=max_seq_length,
        revision=revision,
        length_cache=length_cache,
    )


def engine_from_backends(
    backends: Sequence[EmbeddingBackend],
    devices: Sequence[DeviceInfo],
    settings: Settings,
) -> Engine:
    """Wire built backends into an Engine.

    Tokenizer-based lengths: `max_batch_tokens` then means real tokens, not
    characters. All backends serve the same model with the same tokenizer,
    so the first one's `length_fn` serves the whole engine.
    """
    length_fn = getattr(backends[0], "length_fn", len)
    return Engine(backends, devices, settings, length_fn=length_fn)
