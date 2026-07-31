"""FastAPI app factory. The engine is injected so tests pass a fake."""

from __future__ import annotations

import base64
import logging
import secrets
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from embedx import __version__
from embedx.api.errors import error_body, install_error_handlers
from embedx.api.schemas import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbedRequest,
    Usage,
)
from embedx.config import Settings
from embedx.engine.engine import Engine

logger = logging.getLogger("embedx.api")


class _BodyLimitMiddleware:
    """Reject oversized request bodies.

    Content-Length is the fast path: rejected before any parsing. A request
    without one (chunked transfer) would bypass the cap entirely, so its
    receive channel is wrapped to count bytes as the endpoint consumes the
    stream — the body is never buffered here.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            if raw_length.isdigit() and int(raw_length) > self.max_bytes:
                response = JSONResponse(
                    status_code=413,
                    content=error_body(f"request body exceeds max_request_bytes={self.max_bytes}"),
                )
                await response(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        received = 0

        async def counting_receive() -> dict[str, Any]:
            nonlocal received
            message: dict[str, Any] = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # HTTPException on purpose: FastAPI's body-read guard
                    # re-raises HTTPException untouched but converts any
                    # other exception into a generic 400.
                    raise HTTPException(
                        status_code=413,
                        detail=f"request body exceeds max_request_bytes={self.max_bytes}",
                    )
            return message

        await self.app(scope, counting_receive, send)


# Module-level on purpose: with `from __future__ import annotations` the
# Annotated[..., Depends(_bearer)] annotation below is evaluated lazily
# against module globals, so a closure-local instance would not resolve.
_bearer = HTTPBearer(auto_error=False)


def create_app(settings: Settings, engine: Engine) -> FastAPI:
    """Build the HTTP layer over an already-constructed engine.

    The CLI (task 09) is responsible for building the real engine; here it
    is handed in, so tests inject a deterministic fake.
    """
    app = FastAPI(title="embedx", version=__version__)
    install_error_handlers(app)

    # A wrong pooling produces plausible garbage with no error anywhere
    # else, so the choice is logged where it cannot be missed.
    logger.info(
        "embedx serving model=%s pooling=%s normalize=%s dtype=%s wrapping=%r",
        settings.model_id,
        settings.pooling.value,
        settings.normalize,
        settings.dtype.value,
        settings.wrapping,
    )
    if settings.is_exposed_without_auth:
        logger.warning(settings.exposure_warning())

    async def require_auth(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    ) -> None:
        if settings.api_key is None:
            return
        if credentials is None or not secrets.compare_digest(
            credentials.credentials, settings.api_key
        ):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    app.add_middleware(_BodyLimitMiddleware, max_bytes=settings.max_request_bytes)

    def _as_texts(value: str | list[str]) -> list[str]:
        texts = [value] if isinstance(value, str) else value
        if len(texts) > settings.max_request_items:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"too many inputs: {len(texts)} exceeds "
                    f"max_request_items={settings.max_request_items}"
                ),
            )
        # Wrapping happens here, at the boundary, so that everything
        # downstream sees the final string: length_fn feeds both the
        # per-device token budgets and usage.prompt_tokens, and measuring
        # the caller's raw input would under-count both. Unset wrapping
        # returns the same list object untouched.
        if settings.wrapping is None:
            return texts
        return [settings.wrap(text) for text in texts]

    @app.post("/v1/embeddings", dependencies=[Depends(require_auth)])
    async def embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
        texts = _as_texts(request.input)
        vectors = await run_in_threadpool(engine.embed, texts)
        if request.dimensions is not None and request.dimensions != vectors.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"dimensions={request.dimensions} does not match the model "
                    f"output width {vectors.shape[1]}; embedx does not truncate vectors"
                ),
            )
        data: list[EmbeddingObject] = []
        for index, vector in enumerate(vectors):
            embedding: list[float] | str
            if request.encoding_format == "base64":
                # Little-endian float32 bytes, one string per embedding —
                # what the official OpenAI client decodes by default.
                embedding = base64.b64encode(
                    np.ascontiguousarray(vector, dtype="<f4").tobytes()
                ).decode("ascii")
            else:
                embedding = vector.tolist()
            data.append(EmbeddingObject(index=index, embedding=embedding))
        # Engine.token_count measures in the engine's own unit: real tokens
        # on the GPU factory path, characters with the default `len`.
        total = await run_in_threadpool(engine.token_count, texts)
        return EmbeddingsResponse(
            data=data,
            model=request.model,
            usage=Usage(prompt_tokens=total, total_tokens=total),
        )

    @app.post("/embed", dependencies=[Depends(require_auth)])
    async def embed(request: EmbedRequest) -> list[list[float]]:
        texts = _as_texts(request.inputs)
        vectors = await run_in_threadpool(engine.embed, texts)
        result: list[list[float]] = vectors.tolist()
        return result

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Open (no auth) and touches no GPU state.
        return {"status": "ok"}

    @app.get("/info", dependencies=[Depends(require_auth)])
    async def info() -> dict[str, Any]:
        return {
            "model_id": settings.model_id,
            "pooling": settings.pooling.value,
            "normalize": settings.normalize,
            "dtype": settings.dtype.value,
            # Verbatim, so a caller can see exactly what is prepended to
            # their text without reading the server's configuration.
            "wrapping": settings.wrapping,
            "version": __version__,
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "weight": device.weight,
                    "max_batch_tokens": budget,
                    # Silent truncation must be visible somewhere in prod.
                    "truncated_count": truncated,
                }
                for (device, budget), truncated in zip(
                    engine.devices_with_budgets, engine.truncated_counts, strict=True
                )
            ],
        }

    return app
