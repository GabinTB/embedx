"""FastAPI app factory. The model source is injected so tests pass a fake."""

from __future__ import annotations

import base64
import logging
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Annotated, Any, Protocol, TypeVar

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
from embedx.config import Dtype, Pooling, Settings
from embedx.engine.engine import Engine
from embedx.registry import (
    DEFAULT_KEEP_ALIVE_S,
    ModelPlacementError,
    ModelStatus,
    PoolingConflictError,
    PoolingRequiredError,
    RegistryError,
    UnsupportedWeightFormatError,
    WeightFormatUnverifiableError,
)

logger = logging.getLogger("embedx.api")

T = TypeVar("T")

# Registry failures the caller can act on. Ordered most-specific first and
# matched with isinstance, so a future subclass maps with its parent rather
# than falling through to a 500. Anything not listed here — including a bare
# RegistryError — reaches the unhandled-exception handler and returns a
# generic body with the traceback logged server-side only.
_REGISTRY_ERROR_STATUS: tuple[tuple[type[RegistryError], int], ...] = (
    # The request is incomplete: a first load with no pooling to resolve.
    (PoolingRequiredError, 400),
    # Not a malformed request — a state conflict with a resident model.
    (PoolingConflictError, 409),
    # The checkpoint is refused, permanently. Retrying will not help.
    (UnsupportedWeightFormatError, 400),
    # SPEC-GAP: task 14 assigns this 400 alongside the line above, though it
    # is the transient one — the hub was unreachable and the model was not
    # cached, so a retry may well succeed where 4xx implies it will not.
    (WeightFormatUnverifiableError, 400),
    # The server cannot serve this model right now; the request was fine.
    (ModelPlacementError, 503),
)


class ModelSource(Protocol):
    """What the HTTP layer needs from a model registry.

    A Protocol rather than `ModelRegistry` itself so the legacy adapter
    below, and test doubles, satisfy it structurally.
    """

    def acquire(
        self,
        model_id: str,
        pooling: Pooling | None = None,
        dtype: Dtype = Dtype.AUTO,
        max_seq_len: int | None = None,
        keep_alive: float | None = None,
    ) -> AbstractContextManager[Engine]: ...

    def list_loaded(self) -> list[ModelStatus]: ...


class _FixedModelSource:
    """The 0.1.0 single-model path, behind the registry interface.

    `serve` still builds one Engine eagerly and passes it here; task 15
    deletes that call and this class with it. Until then this adapter
    exists so the routes have exactly one shape instead of an
    `if registry is not None` in every endpoint.

    Behaviour matches 0.1.0: `model` was an echoed label, not a routing
    key, so any model id is served by the one configured engine. A pooling
    that disagrees with the configured one is the single exception — that
    is conflict-checked, because answering with the wrong pooling is the
    failure this project refuses to make quietly.
    """

    def __init__(self, settings: Settings, engine: Engine) -> None:
        self._settings = settings
        self._engine = engine
        self._loaded_at = time.time()

    @contextmanager
    def acquire(
        self,
        model_id: str,
        pooling: Pooling | None = None,
        dtype: Dtype = Dtype.AUTO,
        max_seq_len: int | None = None,
        keep_alive: float | None = None,
    ) -> Iterator[Engine]:
        if pooling is not None and pooling is not self._settings.pooling:
            raise PoolingConflictError(model_id, pooling, self._settings.pooling)
        yield self._engine

    def list_loaded(self) -> list[ModelStatus]:
        return [
            ModelStatus(
                model_id=self._settings.model_id,
                pooling=self._settings.pooling,
                dtype=self._settings.dtype,
                device_indices=tuple(
                    device.index for device, _ in self._engine.devices_with_budgets
                ),
                # Never idle and never referenced: it is pinned for the
                # process lifetime, not managed by a reaper.
                idle_s=0.0,
                last_used_epoch_s=self._loaded_at,
                ref_count=0,
            )
        ]


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


def create_app(
    settings: Settings,
    engine: Engine | None = None,
    *,
    registry: ModelSource | None = None,
) -> FastAPI:
    """Build the HTTP layer over a model registry.

    `registry` is the real path: models resolve per request and load on
    demand, so it is constructible with nothing loaded. `engine` is the
    0.1.0 single-model path that `serve` still uses until task 15 removes
    it; passing one wraps it in `_FixedModelSource`. Exactly one is needed.
    """
    if (engine is None) == (registry is None):
        raise ValueError("create_app needs exactly one of `engine` or `registry`")
    source: ModelSource = registry if registry is not None else _FixedModelSource(settings, engine)  # type: ignore[arg-type]
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

    def _use_model(request: EmbeddingsRequest | EmbedRequest, work: Callable[[Engine], T]) -> T:
        # Runs entirely on the worker thread. acquire, embed and release are
        # one unit here on purpose: acquiring on the event loop and
        # releasing on a worker (or the reverse) would leave the reference
        # count describing something other than "a request is using this".
        # The `with` wraps the inference, not just the lookup.
        with source.acquire(
            request.model,
            pooling=request.pooling,
            # The registry's own default is AUTO; the request says "unset".
            dtype=request.dtype if request.dtype is not None else Dtype.AUTO,
            max_seq_len=request.max_seq_len,
            keep_alive=request.keep_alive,
        ) as model_engine:
            return work(model_engine)

    async def _run(request: EmbeddingsRequest | EmbedRequest, work: Callable[[Engine], T]) -> T:
        try:
            return await run_in_threadpool(_use_model, request, work)
        except RegistryError as exc:
            for error_type, status in _REGISTRY_ERROR_STATUS:
                if isinstance(exc, error_type):
                    # A load failure is a normal outcome, not a crash: it
                    # goes through the same envelope as any other 4xx/5xx,
                    # with no traceback in the body.
                    logger.warning("%s for model %s: %s", type(exc).__name__, request.model, exc)
                    raise HTTPException(status_code=status, detail=str(exc)) from exc
            raise  # unmapped: generic 500 body, traceback logged server-side

    @app.post(
        "/v1/embeddings",
        dependencies=[Depends(require_auth)],
        description=(
            "OpenAI-compatible embeddings. `model` is resolved live: if the "
            "named model is not resident, this request loads it and BLOCKS "
            "until the load finishes, which for a cold model can take a while. "
            "There is no separate asynchronous pull endpoint in this release. "
            "`pooling` is required the first time a model is loaded."
        ),
    )
    async def embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
        texts = _as_texts(request.input)
        # Both measured by the engine that served them, inside one
        # acquisition: usage must come from the same tokenizer that batched.
        vectors, total = await _run(
            request, lambda served: (served.embed(texts), served.token_count(texts))
        )
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
        return EmbeddingsResponse(
            data=data,
            model=request.model,
            usage=Usage(prompt_tokens=total, total_tokens=total),
        )

    @app.post(
        "/embed",
        dependencies=[Depends(require_auth)],
        description=(
            "TEI-style embeddings, returning bare vectors. Unlike TEI, `model` "
            "is required — embedx routes per request. A model that is not "
            "resident is loaded now and this request BLOCKS until the load "
            "finishes; there is no separate asynchronous pull endpoint."
        ),
    )
    async def embed(request: EmbedRequest) -> list[list[float]]:
        texts = _as_texts(request.inputs)
        vectors = await _run(request, lambda served: served.embed(texts))
        result: list[list[float]] = vectors.tolist()
        return result

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Open (no auth) and touches no GPU state.
        return {"status": "ok"}

    @app.get(
        "/info",
        dependencies=[Depends(require_auth)],
        description=(
            "Server configuration and every currently resident model. An "
            "empty `models` list is normal — a freshly started server has "
            "loaded nothing. Reading this does not keep any model alive."
        ),
    )
    async def info() -> dict[str, Any]:
        # list_loaded, never acquire: introspection must not take a
        # reference, or polling /info would pin every model against the
        # reaper forever.
        return {
            "version": __version__,
            "normalize": settings.normalize,
            # Verbatim, so a caller can see exactly what is prepended to
            # their text without reading the server's configuration.
            "wrapping": settings.wrapping,
            # SPEC-GAP: hardcoded until task 15 adds
            # Settings.default_keep_alive_s and threads the real value here.
            "default_keep_alive_s": DEFAULT_KEEP_ALIVE_S,
            "models": [
                {
                    "model_id": status.model_id,
                    "pooling": status.pooling.value,
                    "dtype": status.dtype.value,
                    "devices": list(status.device_indices),
                    "idle_s": status.idle_s,
                    "last_used_epoch_s": status.last_used_epoch_s,
                    "ref_count": status.ref_count,
                }
                for status in source.list_loaded()
            ],
        }

    return app
