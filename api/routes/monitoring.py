"""Monitoring endpoints for the inference API."""

from __future__ import annotations

from fastapi import APIRouter

from api.state import state

router = APIRouter(prefix="/api/v1", tags=["monitoring"])


@router.get("/monitoring/status")
def monitoring_status() -> dict:
    """Returns a lightweight monitoring status payload."""
    return {
        "model_ready": state.model_ready,
        "predictor_loaded": state.predictor is not None,
        "artifacts_loaded": state.artifacts is not None,
        "model_version": state.artifacts.model_version if state.artifacts else "unavailable",
    }
