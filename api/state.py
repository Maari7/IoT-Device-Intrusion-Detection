"""Shared runtime state for the inference API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.inference.model_loader import InferenceArtifacts
from src.inference.predictor import HybridPredictor


@dataclass
class AppState:
    """Holds process-wide inference state."""

    artifacts: Optional[InferenceArtifacts] = None
    predictor: Optional[HybridPredictor] = None
    model_ready: bool = False


state = AppState()
