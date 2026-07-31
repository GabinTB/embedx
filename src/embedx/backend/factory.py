"""Engine factory: device discovery → ranking → one HFBackend per device.

Importable without torch; the HFBackend import happens inside
`build_engine`, which is the first point where torch is genuinely needed.
"""

from __future__ import annotations

from embedx.config import Settings
from embedx.engine.engine import Engine
from embedx.gpu.discovery import discover_devices, rank_devices


def build_engine(settings: Settings) -> Engine:
    """Assemble the real engine: one full model copy per ranked device.

    Data sharding, not model parallelism — every device holds the whole
    model and the scheduler splits the inputs.
    """
    infos = discover_devices(settings.devices)
    if not infos:
        raise RuntimeError(
            "no CUDA device found: embedx needs torch with CUDA available "
            "(install the gpu extra on a CUDA host: uv sync --extra gpu); "
            "CPU serving is not supported"
        )
    ranked = rank_devices(infos, settings.device_weights)

    from embedx.backend.hf import HFBackend  # lazy: pulls in torch

    backends = [
        HFBackend(
            model_id=settings.model_id,
            device_index=device.index,
            pooling=settings.pooling,
            normalize=settings.normalize,
            dtype=settings.dtype,
            max_seq_length=settings.max_seq_len,
            revision=settings.revision,
        )
        for device in ranked
    ]
    # Tokenizer-based lengths: max_batch_tokens then means real tokens, not
    # characters. All backends share one tokenizer, so the first one serves.
    return Engine(backends, ranked, settings, length_fn=backends[0].length_fn)
