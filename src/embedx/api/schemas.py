"""OpenAI-compatible request/response models, plus the TEI-style /embed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


def _validate_text_input(value: object) -> object:
    """Shared input checks: non-empty, text only.

    OpenAI's API also accepts pre-tokenized input (`list[int]` or
    `list[list[int]]`); embedx does not, and the rejection must say so
    rather than surfacing a cryptic union error.
    """
    if isinstance(value, list):
        if not value:
            raise ValueError("input must not be an empty list")
        if all(isinstance(item, int) for item in value) or all(
            isinstance(item, list) and item and all(isinstance(t, int) for t in item)
            for item in value
        ):
            raise ValueError(
                "embedx does not accept pre-tokenized input (token arrays); "
                "send text as a string or a list of strings"
            )
    return value


class EmbeddingsRequest(BaseModel):
    """POST /v1/embeddings body, mirroring OpenAI."""

    input: str | list[str]
    model: str
    # Not optional to implement: the official OpenAI Python client sends
    # base64 by default and decodes client-side, so a float-only server
    # silently breaks the most common caller.
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None

    @field_validator("input", mode="before")
    @classmethod
    def _check_input(cls, value: object) -> object:
        return _validate_text_input(value)


class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str  # str carries the base64 form


class Usage(BaseModel):
    # Counts come from Engine.token_count, i.e. the same length measure the
    # engine batches with: real token counts when the tokenizer-based
    # length_fn is injected (the GPU factory path), plain character counts
    # with the default `len` (fakes and CPU tests) — correct for those,
    # since no tokenizer exists there.
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: Usage


class EmbedRequest(BaseModel):
    """POST /embed body (TEI-style); the response is a bare list of vectors."""

    inputs: str | list[str]

    @field_validator("inputs", mode="before")
    @classmethod
    def _check_inputs(cls, value: object) -> object:
        return _validate_text_input(value)
