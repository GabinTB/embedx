"""Real Hugging Face GPU backend — the `gpu` extra.

torch / transformers / sentence-transformers are imported lazily inside
functions so `import embedx.backend.hf` works with torch absent; the
task-06 ast guard enforces this. Torch types may appear in annotations
only under TYPE_CHECKING.

Vendor-specific facts — the device string and which dtypes the hardware
supports — come from an `Accelerator` rather than being spelled `cuda`
here. torch itself is still imported directly: the tensor library is not
what the seam abstracts, the *device runtime* is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from embedx.config import Dtype, Pooling
from embedx.gpu.vendor import Accelerator, DeviceInfo, get_accelerator

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
    except Exception as exc:
        # Not silent: a cached ST model with an unreachable hub would be
        # served through the plain path, and the operator must know that.
        logger.warning(
            "could not determine whether %s is a sentence-transformers "
            "checkpoint (%s: %s); using the plain AutoModel path with the "
            "configured pooling",
            model_id,
            type(exc).__name__,
            exc,
        )
        return False


def _resolve_dtype(dtype: Dtype, info: DeviceInfo, accelerator: Accelerator, torch_mod: Any) -> Any:
    if dtype is Dtype.FLOAT32:
        return torch_mod.float32
    if dtype is Dtype.FLOAT16:
        return torch_mod.float16
    if dtype is Dtype.BFLOAT16:
        return torch_mod.bfloat16
    # AUTO: which reduced precisions the hardware actually supports is a
    # vendor fact, not a universal one — `capability major >= 8` is true of
    # CUDA and meaningless on ROCm — so the accelerator answers it.
    if accelerator.supports_bfloat16(info):
        return torch_mod.bfloat16
    if accelerator.supports_float16(info):
        return torch_mod.float16
    return torch_mod.float32


class TokenLengthCache:
    """text -> unclamped token length, shared by every device's backend.

    The keys are RAW INPUT TEXT, so the bound is a retention decision as
    much as a memory one: a long-lived server must not accumulate unbounded
    user input. Eviction is a full clear — crude on purpose, because
    repeats matter within a request, not across the process lifetime — and
    triggers on whichever cap is hit first: entry count, or total key bytes
    (65536 document-length entries would retain gigabytes).
    """

    def __init__(self, max_entries: int = 65536, max_bytes: int = 16_000_000) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: dict[str, int] = {}
        self._key_bytes = 0

    def get(self, text: str) -> int | None:
        return self._entries.get(text)

    def put(self, text: str, length: int) -> None:
        if text in self._entries:
            return
        size = len(text.encode("utf-8"))
        if len(self._entries) >= self.max_entries or self._key_bytes + size > self.max_bytes:
            self._entries.clear()
            self._key_bytes = 0
        self._entries[text] = length
        self._key_bytes += size


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
        length_cache: TokenLengthCache | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        torch_mod = _import_torch()
        self._torch = torch_mod
        self._accelerator = accelerator if accelerator is not None else get_accelerator()
        self.model_id = model_id
        self.device_index = device_index
        self.pooling = pooling
        self.normalize = normalize
        self.revision = revision
        self.truncated_count = 0
        self.dim: int = 0
        self.max_seq_length: int = 0
        # The caller passes one cache shared by all of a model's devices (see
        # the registry); a private instance keeps direct construction working.
        self._length_cache = length_cache if length_cache is not None else TokenLengthCache()
        self._tokenizer: Any = None
        self._st_model: Any = None
        self._model: Any = None

        info = self._accelerator.device_info(device_index)
        self._device = torch_mod.device(self._accelerator.device_string(device_index))
        self._dtype = _resolve_dtype(dtype, info, self._accelerator, torch_mod)
        logger.info(
            "device %d: dtype %s (configured %s, capability major %d)",
            device_index,
            self._dtype,
            dtype.value,
            info.capability[0],
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
        # A text is truncated exactly when its unclamped token length
        # exceeds the clamp — read from the cache the batcher's length_fn
        # already filled, so no re-tokenization happens here.
        truncated = sum(1 for text in texts if self._raw_token_length(text) > self.max_seq_length)
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

    def _raw_token_length(self, text: str) -> int:
        """Unclamped token length via the engine-wide shared cache.

        The shared cache is what keeps each text to ONE tokenization per
        request: the batcher calls `backends[0].length_fn`, and every other
        device's `_count_truncations` then reads the same entries instead
        of re-tokenizing the batch cold. All backends serve the same model
        with the same tokenizer, so entries are valid for all of them.
        """
        cached = self._length_cache.get(text)
        if cached is not None:
            return cached
        length = len(self._tokenizer(text, truncation=False, padding=False)["input_ids"])
        self._length_cache.put(text, length)
        return length

    def length_fn(self, text: str) -> int:
        """True token length, clamped to `max_seq_length`, for the batcher.

        No padding, no tensors. The clamp matters because an over-long text
        is truncated by `embed`, so its real padded cost is
        `max_seq_length`, not its raw length.
        """
        return min(self._raw_token_length(text), self.max_seq_length)
