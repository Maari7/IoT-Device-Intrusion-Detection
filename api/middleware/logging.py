"""Request logging middleware for the inference API."""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

LOGGER = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs request lifecycle and latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Logs request/response metadata."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        LOGGER.info(
            "request method=%s path=%s status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
