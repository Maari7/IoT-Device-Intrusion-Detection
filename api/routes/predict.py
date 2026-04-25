"""Prediction endpoints for the hybrid IDS API."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from api.metrics import APP_LATENCY, APP_REQUESTS
from api.schemas.request import BatchPredictionRequest, PredictionRequest
from api.schemas.response import BatchPredictionItem, BatchPredictionResponse, PredictionResponse
from api.state import state

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])


@router.post("/single", response_model=PredictionResponse)
def predict_single(request: PredictionRequest) -> PredictionResponse:
    """Predicts a single sample."""
    if not state.model_ready or state.predictor is None or state.artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    with APP_LATENCY.labels(endpoint="single").time():
        result = state.predictor.predict_single(request.features)

    APP_REQUESTS.labels(endpoint="single", status="200").inc()
    return PredictionResponse(
        prediction=result.prediction,
        label=result.label,
        confidence=result.confidence,
        anomaly_score=result.anomaly_score,
        model_version=state.artifacts.model_version,
        details=result.details,
    )


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predicts a batch of samples."""
    if not state.model_ready or state.predictor is None or state.artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    with APP_LATENCY.labels(endpoint="batch").time():
        results = state.predictor.predict_batch(request.items)

    APP_REQUESTS.labels(endpoint="batch", status="200").inc()
    items = [
        BatchPredictionItem(
            prediction=result.prediction,
            label=result.label,
            confidence=result.confidence,
            anomaly_score=result.anomaly_score,
        )
        for result in results
    ]
    return BatchPredictionResponse(
        items=items,
        model_version=state.artifacts.model_version,
        batch_size=len(items),
    )
