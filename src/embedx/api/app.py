"""FastAPI app factory. The engine is injected so tests pass a fake."""

from __future__ import annotations

import base64
import logging
import secrets
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from embedx import __version__
from embedx.api.errors import error_body, install_error_handlers
from embedx.api.schemas import EmbeddingsRequest, EmbedRequest
from embedx.config import Settings
from embedx.engine.engine import Engine

logger = logging.getLogger("embedx.api")

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
        "embedx serving model=%s pooling=%s normalize=%s dtype=%s",
        settings.model_id,
        settings.pooling.value,
        settings.normalize,
        settings.dtype.value,
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

    @app.middleware("http")
    async def limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Reject oversized bodies on Content-Length, before any parsing.
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > settings.max_request_bytes:
            return JSONResponse(
                status_code=413,
                content=error_body(
                    f"request body exceeds max_request_bytes={settings.max_request_bytes}"
                ),
            )
        return await call_next(request)

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
        return texts

    @app.post("/v1/embeddings", dependencies=[Depends(require_auth)])
    async def embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
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
        data = []
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
            data.append({"object": "embedding", "index": index, "embedding": embedding})
        # Character-based approximation (the same length measure the engine
        # batches with) until task 08 supplies a real tokenizer.
        chars = sum(len(text) for text in texts)
        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {"prompt_tokens": chars, "total_tokens": chars},
        }

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
            "version": __version__,
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "weight": device.weight,
                    "max_batch_tokens": budget,
                }
                for device, budget in engine.devices_with_budgets
            ],
        }

    return app
