"""Health endpoints for the hybrid IDS API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.response import HealthResponse, ModelInfoResponse
from api.state import state

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Returns overall service health."""
    return HealthResponse(
        status="ok" if state.model_ready else "starting",
        model_ready=state.model_ready,
        details={"predictor": str(state.predictor is not None), "artifacts": str(state.artifacts is not None)},
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Returns metadata for the currently loaded model."""
    if not state.model_ready or state.artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    return ModelInfoResponse(
        model_version=state.artifacts.model_version,
        champion_source=state.artifacts.champion_source,
        artifact_dir=str(state.artifacts.artifact_dir),
        thresholds={
            "decision_threshold": 0.5,
        },
        status="ready",
    )
