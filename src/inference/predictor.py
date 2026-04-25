"""Hybrid prediction logic for real-time and batch inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.inference.model_loader import InferenceArtifacts

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured prediction output."""

    prediction: int
    label: str
    confidence: float
    anomaly_score: float
    details: Dict[str, float]


class HybridPredictor:
    """Produces hybrid predictions from baseline, SHAP, autoencoder, and fusion artifacts."""

    def __init__(self, artifacts: InferenceArtifacts, decision_threshold: float = 0.5) -> None:
        self.artifacts = artifacts
        self.decision_threshold = decision_threshold
        self.feature_columns = artifacts.feature_columns

    def _prepare_frame(self, features: Dict[str, float]) -> pd.DataFrame:
        """Normalizes a single feature dict into a one-row DataFrame."""
        missing = [col for col in self.feature_columns if col not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing[:5]}")
        ordered = {column: float(features[column]) for column in self.feature_columns}
        return pd.DataFrame([ordered], columns=self.feature_columns)

    def _transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Applies the saved Phase 1 preprocessing artifacts to features only."""
        payload = self.artifacts.preprocessor
        imputer = payload["imputer"]
        scaler = payload["scaler"]
        lower_bounds = payload["lower_bounds"]
        upper_bounds = payload["upper_bounds"]
        columns = payload["feature_columns"]

        x = frame[columns].astype("float32")
        imputed = pd.DataFrame(imputer.transform(x), columns=columns, index=frame.index)
        clipped = imputed.clip(lower_bounds, upper_bounds, axis=1)
        if scaler is not None:
            scaled = scaler.transform(clipped)
            return pd.DataFrame(scaled, columns=columns, index=frame.index)
        return clipped

    def _infer_benign_class_id(self, baseline_model: Any) -> int:
        """Infers benign class index from the baseline estimator classes if possible."""
        classes = getattr(baseline_model, "classes_", None)
        if classes is None:
            return 0
        if len(classes) == 0:
            return 0
        return 0

    def predict_single(self, features: Dict[str, float]) -> PredictionResult:
        """Runs a single hybrid prediction."""
        frame = self._prepare_frame(features)
        x = self._transform_features(frame)

        baseline_model = self.artifacts.baseline_model
        fusion_model = self.artifacts.fusion_model
        autoencoder = self.artifacts.autoencoder
        shap_weights = self.artifacts.shap_weights
        dual_scorer = self.artifacts.dual_consistency

        weighted = shap_weights.transform(x)
        reconstruction_error = autoencoder.reconstruction_error(weighted)
        baseline_proba = baseline_model.predict_proba(x)
        benign_class_id = self._infer_benign_class_id(baseline_model)
        classifier_attack_score = 1.0 - baseline_proba[:, benign_class_id]
        dual_score = dual_scorer.score(classifier_attack_score, reconstruction_error)

        fusion_matrix = np.column_stack([classifier_attack_score, reconstruction_error, dual_score])
        fusion_proba = fusion_model.predict_proba(fusion_matrix)[:, 1]
        fusion_pred = fusion_model.predict(fusion_matrix)

        is_attack = int(fusion_pred[0])
        confidence = float(fusion_proba[0])
        label = "attack" if is_attack == 1 else "benign"

        details = {
            "baseline_attack_score": float(classifier_attack_score[0]),
            "reconstruction_error": float(reconstruction_error[0]),
            "dual_score": float(dual_score[0]),
            "fusion_probability": confidence,
        }

        return PredictionResult(
            prediction=is_attack,
            label=label,
            confidence=confidence,
            anomaly_score=float(dual_score[0]),
            details=details,
        )

    def predict_batch(self, items: List[Dict[str, float]]) -> List[PredictionResult]:
        """Runs a batch of predictions."""
        results: List[PredictionResult] = []
        for item in items:
            results.append(self.predict_single(item))
        return results
