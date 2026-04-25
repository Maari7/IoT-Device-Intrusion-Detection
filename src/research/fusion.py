"""Fusion model for combining classifier and anomaly signals."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


class FusionModel:
    """Binary fusion model that learns from stacked research signals."""

    def __init__(self, config: Dict | None = None) -> None:
        cfg = config or {}
        self.model = LogisticRegression(
            C=float(cfg.get("c", 1.0)),
            max_iter=int(cfg.get("max_iter", 1000)),
            class_weight=cfg.get("class_weight", "balanced"),
            solver=cfg.get("solver", "lbfgs"),
            random_state=int(cfg.get("random_state", 42)),
        )

    def fit(self, signal_matrix: np.ndarray, y_binary: np.ndarray) -> "FusionModel":
        """Fits logistic regression over stacked hybrid signals."""
        self.model.fit(signal_matrix, y_binary)
        return self

    def predict(self, signal_matrix: np.ndarray) -> np.ndarray:
        """Predicts binary class labels."""
        return self.model.predict(signal_matrix)

    def predict_proba(self, signal_matrix: np.ndarray) -> np.ndarray:
        """Predicts binary class probabilities."""
        return self.model.predict_proba(signal_matrix)

    def save(self, output_path: str | Path) -> Path:
        """Serializes model to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        LOGGER.info("Saved fusion model to %s", path)
        return path

    @staticmethod
    def save_signal_metadata(output_path: str | Path, metadata: Dict) -> Path:
        """Writes signal-level metadata to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path
