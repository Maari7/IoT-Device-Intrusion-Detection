"""Shared Prometheus metrics for the inference API."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

APP_REQUESTS = Counter("hx_idl_requests_total", "Total API requests", ["endpoint", "status"])
APP_LATENCY = Histogram("hx_idl_request_latency_seconds", "Request latency", ["endpoint"])
