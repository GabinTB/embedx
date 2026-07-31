"""One OpenAI-shaped error envelope for every failure path."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("embedx.api")


def error_body(
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        message = str(first.get("msg", "invalid request"))
        loc = [str(part) for part in first.get("loc", ()) if part != "body"]
        return JSONResponse(
            status_code=400,
            content=error_body(message, param=".".join(loc) or None),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
        code = "invalid_api_key" if exc.status_code == 401 else None
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(str(exc.detail), error_type, code=code),
        )

    @app.exception_handler(Exception)
    async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        # Full traceback server-side; a generic message client-side. Never
        # leak internals into the response body.
        logger.exception("unhandled error serving %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body("internal server error", "server_error"),
        )
