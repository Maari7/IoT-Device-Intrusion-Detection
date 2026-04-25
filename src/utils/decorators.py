"""Reusable decorators for retries and execution timing."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from src.utils.logger import get_logger

F = TypeVar("F", bound=Callable[..., Any])
LOGGER = get_logger(__name__)


def retry(attempts: int = 3, delay_seconds: float = 1.0) -> Callable[[F], F]:
    """Retries a callable when it raises an exception."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - defensive behavior
                    last_error = exc
                    LOGGER.warning(
                        "retry attempt=%d/%d function=%s error=%s",
                        attempt,
                        attempts,
                        func.__name__,
                        str(exc),
                    )
                    if attempt < attempts:
                        time.sleep(delay_seconds)
            if last_error is not None:
                raise last_error
            raise RuntimeError("retry wrapper reached unexpected state")

        return wrapper  # type: ignore[return-value]

    return decorator


def timed(func: F) -> F:
    """Logs execution time of a callable."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        LOGGER.info("timed function=%s duration_seconds=%.4f", func.__name__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]
