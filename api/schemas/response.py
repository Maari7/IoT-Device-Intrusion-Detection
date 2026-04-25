"""Response models for hybrid inference API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Single prediction response."""

    prediction: int = Field(..., description="Predicted class label or anomaly flag")
    label: str = Field(..., description="Human-readable prediction label")
    confidence: float = Field(..., description="Prediction confidence")
    anomaly_score: float = Field(..., description="Hybrid anomaly score")
    model_version: str = Field(..., description="Model version used")
    details: Dict[str, float] = Field(default_factory=dict)


class BatchPredictionItem(BaseModel):
    """One item in a batch prediction response."""

    prediction: int
    label: str
    confidence: float
    anomaly_score: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    items: List[BatchPredictionItem]
    model_version: str
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    model_ready: bool
    details: Dict[str, str] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    """Model information response model."""

    model_version: str
    champion_source: str
    artifact_dir: str
    thresholds: Dict[str, float]
    status: str
