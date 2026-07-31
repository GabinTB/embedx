"""Backend protocol: the seam that lets core logic run without torch.

Any embedding backend (real GPU-backed or fake) satisfies this protocol
structurally; core modules depend only on it and never import torch.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """A synchronous text-embedding backend.

    Implementations must set `dim` and return arrays of shape
    `(len(texts), dim)` from `embed`, preserving input order.
    """

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed `texts`, returning an array of shape `(len(texts), dim)`."""
        ...
