"""Error handling middleware helpers for the inference API."""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

LOGGER = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Converts unexpected exceptions into consistent JSON errors."""

    async def dispatch(self, request: Request, call_next: Callable):
        """Wraps request processing in error handling."""
        try:
            return await call_next(request)
        except Exception as exc:  # pragma: no cover - safety net
            LOGGER.exception("Unhandled API error on %s %s", request.method, request.url.path)
            return JSONResponse(status_code=500, content={"detail": str(exc)})
