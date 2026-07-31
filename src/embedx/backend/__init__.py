"""Embedding backends. Core code imports only the protocol and fakes here;
torch-backed backends are imported lazily behind the `gpu` extra."""

from embedx.backend.base import EmbeddingBackend
from embedx.backend.fake import FakeBackend

__all__ = ["EmbeddingBackend", "FakeBackend"]
