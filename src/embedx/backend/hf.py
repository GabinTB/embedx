"""Real Hugging Face CUDA backend — the `gpu` extra.

torch / transformers / sentence-transformers are imported lazily inside
functions so `import embedx.backend.hf` works with torch absent; the
task-06 ast guard enforces this. Torch types may appear in annotations
only under TYPE_CHECKING.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from embedx.config import Dtype, Pooling

if TYPE_CHECKING:  # annotation-only; never executes at runtime
    import torch

logger = logging.getLogger("embedx.backend.hf")


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "HFBackend requires torch: install the gpu extra (uv sync --extra gpu)"
        ) from exc
    return torch


def _is_st_checkpoint(model_id: str, revision: str | None) -> bool:
    """True when the checkpoint ships a sentence-transformers modules.json."""
    local = Path(model_id)
    if local.is_dir():
        return (local / "modules.json").is_file()
    try:
        from huggingface_hub import file_exists

        return bool(file_exists(model_id, "modules.json", revision=revision))
    except Exception:  # offline / gated / hub error: fall back to AutoModel
        return False


def _resolve_dtype(dtype: Dtype, capability_major: int, torch_mod: Any) -> Any:
    if dtype is Dtype.FLOAT32:
        return torch_mod.float32
    if dtype is Dtype.FLOAT16:
        return torch_mod.float16
    if dtype is Dtype.BFLOAT16:
        return torch_mod.bfloat16
    # AUTO: bfloat16 needs Ampere (major >= 8); float16 is solid from
    # Volta/Turing (major >= 7); anything older computes in float32.
    if capability_major >= 8:
        return torch_mod.bfloat16
    if capability_major >= 7:
        return torch_mod.float16
    return torch_mod.float32


def _declared_st_pooling(module: Any) -> Pooling | None:
    """Map a sentence-transformers Pooling module's flags to our enum."""
    if getattr(module, "pooling_mode_mean_tokens", False):
        return Pooling.MEAN
    if getattr(module, "pooling_mode_cls_token", False):
        return Pooling.CLS
    if getattr(module, "pooling_mode_lasttoken", False):
        return Pooling.LAST_TOKEN
    return None


class HFBackend:
    """CUDA embedding backend satisfying `EmbeddingBackend`.

    One full model copy per device: the engine shards data across devices,
    not the model. Output is always float32 in input order, whatever the
    compute dtype.
    """

    def __init__(
        self,
        model_id: str,
        device_index: int,
        pooling: Pooling,
        normalize: bool,
        dtype: Dtype,
        max_seq_length: int | None,
        revision: str | None = None,
    ) -> None:
        torch_mod = _import_torch()
        self._torch = torch_mod
        self.model_id = model_id
        self.device_index = device_index
        self.pooling = pooling
        self.normalize = normalize
        self.revision = revision
        self.truncated_count = 0
        self.dim: int = 0
        self.max_seq_length: int = 0
        self._tokenizer: Any = None
        self._st_model: Any = None
        self._model: Any = None

        self._device = torch_mod.device("cuda", device_index)
        major, _minor = torch_mod.cuda.get_device_capability(device_index)
        self._dtype = _resolve_dtype(dtype, major, torch_mod)
        logger.info(
            "device %d: dtype %s (configured %s, capability major %d)",
            device_index,
            self._dtype,
            dtype.value,
            major,
        )

        if _is_st_checkpoint(model_id, revision):
            self._load_sentence_transformers(max_seq_length)
        else:
            self._load_auto_model(max_seq_length)
        logger.info(
            "device %d: %s loaded, dim=%d, pooling=%s, max_seq_length=%d",
            device_index,
            model_id,
            self.dim,
            pooling.value,
            self.max_seq_length,
        )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _load_sentence_transformers(self, max_seq_length: int | None) -> None:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Pooling as STPooling

        model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            device=str(self._device),
            model_kwargs={"torch_dtype": self._dtype},
        )
        model.eval()
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length
        self.max_seq_length = int(model.max_seq_length)
        self._tokenizer = model.tokenizer
        self.dim = int(model.get_sentence_embedding_dimension())

        # The configured pooling always wins, but a disagreement with what
        # the checkpoint declares is exactly the silent-garbage trap, so it
        # is warned about with both values named.
        for module in model:
            if isinstance(module, STPooling):
                declared = _declared_st_pooling(module)
                if declared is not None and declared is not self.pooling:
                    logger.warning(
                        "checkpoint %s declares pooling %s but configuration says %s; "
                        "using the configured %s",
                        self.model_id,
                        declared.value,
                        self.pooling.value,
                        self.pooling.value,
                    )
                module.pooling_mode_mean_tokens = self.pooling is Pooling.MEAN
                module.pooling_mode_cls_token = self.pooling is Pooling.CLS
                module.pooling_mode_lasttoken = self.pooling is Pooling.LAST_TOKEN
                module.pooling_mode_max_tokens = False
                if hasattr(module, "pooling_mode_weightedmean_tokens"):
                    module.pooling_mode_weightedmean_tokens = False
        self._st_model = model

    def _load_auto_model(self, max_seq_length: int | None) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=self._dtype
        )
        self._model = model.to(self._device).eval()
        self.dim = int(self._model.config.hidden_size)
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        else:
            declared = int(getattr(self._tokenizer, "model_max_length", 0) or 0)
            if 0 < declared < 100_000:  # guard the VERY_LARGE_INTEGER sentinel
                self.max_seq_length = declared
            else:
                self.max_seq_length = int(
                    getattr(self._model.config, "max_position_embeddings", 512)
                )
        # Plain checkpoints declare no pooling; nothing to compare against.

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        self._count_truncations(texts)
        if self._st_model is not None:
            vectors = self._st_model.encode(
                texts,
                batch_size=len(texts),  # the engine already sized this batch
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)
        return self._embed_auto(texts)

    def _count_truncations(self, texts: list[str]) -> None:
        # Tensor-free second pass: silent truncation changes results, so it
        # is counted, and the first occurrence is logged.
        encoded = self._tokenizer(texts, truncation=False, padding=False)["input_ids"]
        truncated = sum(1 for ids in encoded if len(ids) > self.max_seq_length)
        if truncated:
            if self.truncated_count == 0:
                logger.warning(
                    "device %d: %d input(s) longer than max_seq_length=%d were truncated "
                    "(further occurrences are counted, not logged)",
                    self.device_index,
                    truncated,
                    self.max_seq_length,
                )
            self.truncated_count += truncated

    def _embed_auto(self, texts: list[str]) -> np.ndarray:
        torch_mod = self._torch
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self._device)
        with torch_mod.inference_mode():
            hidden = self._model(**encoded).last_hidden_state
            pooled = self._pool(hidden, encoded["attention_mask"])
            if self.normalize:
                pooled = torch_mod.nn.functional.normalize(pooled, p=2, dim=1)
        result: np.ndarray = pooled.float().cpu().numpy()
        return result

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        torch_mod = self._torch
        if self.pooling is Pooling.CLS:
            return hidden[:, 0]
        if self.pooling is Pooling.MEAN:
            # Mask-weighted: a plain hidden.mean(dim=1) would average the
            # padding rows in and shift every vector in the batch.
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
        # LAST_TOKEN: last non-padding position per row, from the mask.
        # hidden[:, -1] would be a pad token under right padding, and the
        # padding side varies by tokenizer.
        positions = torch_mod.arange(mask.shape[1], device=mask.device)
        last = (mask * positions).argmax(dim=1)
        return hidden[torch_mod.arange(hidden.shape[0], device=hidden.device), last]

    # ------------------------------------------------------------------ #
    # Length model
    # ------------------------------------------------------------------ #

    def length_fn(self, text: str) -> int:
        """True token length, clamped to `max_seq_length`, for the batcher.

        No padding, no tensors: this runs once per input per request. The
        clamp matters because an over-long text is truncated by `embed`, so
        its real padded cost is `max_seq_length`, not its raw length.
        """
        ids = self._tokenizer(text, truncation=False, padding=False)["input_ids"]
        return min(len(ids), self.max_seq_length)
