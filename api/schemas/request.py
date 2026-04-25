"""Request models for hybrid inference API."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Single inference request containing raw feature values."""

    features: Dict[str, float] = Field(..., description="Feature-name to value mapping")


class BatchPredictionRequest(BaseModel):
    """Batch inference request containing multiple feature dictionaries."""

    items: List[Dict[str, float]] = Field(..., description="List of feature dictionaries")

    @validator("items")
    def validate_items(cls, value: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Ensures the batch is not empty and stays within a practical size."""
        if not value:
            raise ValueError("items cannot be empty")
        if len(value) > 1024:
            raise ValueError("items exceeds maximum batch size")
        return value
