"""OpenAI-compatible request/response models, plus the TEI-style /embed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from embedx.config import Dtype, Pooling

_FIRST_LOAD_ONLY = (
    "Only takes effect the FIRST time this server loads this model. If the "
    "model is already resident, this value is ignored — it does not reload "
    "the model and does not change how it is served."
)


class _LoadOptions(BaseModel):
    """Per-request model-load hints, shared by both embedding endpoints.

    Every field here describes how a model should be loaded, not how this
    request should be served. A model stays loaded across requests, so the
    first request to name it decides these; later ones inherit them.
    """

    pooling: Pooling | None = Field(
        default=None,
        description=(
            "How token vectors are reduced to one vector per input: cls, mean, "
            "or last_token. REQUIRED the first time this server loads a given "
            "model — embedx never guesses it, because the wrong choice returns "
            "plausible-looking vectors that are silently wrong, with no error. "
            "If the model is already loaded, omit it to use whatever it was "
            "loaded with; sending a DIFFERENT value returns 409 Conflict rather "
            "than reloading the model or quietly ignoring you."
        ),
    )
    dtype: Dtype | None = Field(
        default=None,
        description=(
            "Compute dtype: auto, float32, float16, or bfloat16. 'auto' picks "
            "by device capability. " + _FIRST_LOAD_ONLY
        ),
    )
    max_seq_len: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum input length in tokens; longer inputs are truncated. " + _FIRST_LOAD_ONLY
        ),
    )
    keep_alive: float | None = Field(
        default=None,
        description=(
            "Seconds the model stays resident after its last use before it is "
            "unloaded and its GPU memory released. 0 or less unloads it as soon "
            "as this request finishes. Omit for the server default. " + _FIRST_LOAD_ONLY
        ),
    )


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


class EmbeddingsRequest(_LoadOptions):
    """POST /v1/embeddings body, mirroring OpenAI."""

    input: str | list[str]
    model: str = Field(
        description=(
            "Hugging Face model id or local path. This is a live routing key, "
            "not a label: naming a model this server has not loaded yet loads "
            "it now, and the request blocks until that finishes. There is no "
            "separate pull endpoint."
        )
    )
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


class EmbedRequest(_LoadOptions):
    """POST /embed body (TEI-style); the response is a bare list of vectors.

    Deviates from TEI in one way: `model` is required. TEI serves one model
    per process and so has nothing to route on; embedx resolves the model
    per request, and there is no server-wide default to fall back to.
    """

    inputs: str | list[str]
    model: str = Field(
        description=(
            "Hugging Face model id or local path. Required, unlike TEI's "
            "/embed: embedx routes per request and has no single configured "
            "model to fall back to. An unloaded model is loaded now, and the "
            "request blocks until that finishes."
        )
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def _check_inputs(cls, value: object) -> object:
        return _validate_text_input(value)
