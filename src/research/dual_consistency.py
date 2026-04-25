"""Dual-consistency scoring and threshold calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class DualConsistencyScorer:
    """Combines classifier anomaly confidence with reconstruction error."""

    alpha: float = 0.5
    beta: float = 0.5
    threshold_quantile: float = 0.95

    threshold_: float | None = None

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Applies min-max normalization with safe guard."""
        v_min = float(values.min())
        v_max = float(values.max())
        if v_max <= v_min:
            return np.zeros_like(values)
        return (values - v_min) / (v_max - v_min)

    def score(self, classifier_attack_score: np.ndarray, reconstruction_error: np.ndarray) -> np.ndarray:
        """Computes weighted hybrid anomaly score."""
        clf_norm = self._normalize(classifier_attack_score)
        rec_norm = self._normalize(reconstruction_error)
        total_weight = self.alpha + self.beta
        if total_weight <= 0:
            raise ValueError("alpha + beta must be positive")
        return ((self.alpha * clf_norm) + (self.beta * rec_norm)) / total_weight

    def fit_threshold(
        self,
        classifier_attack_score: np.ndarray,
        reconstruction_error: np.ndarray,
    ) -> float:
        """Fits anomaly threshold using score quantile over reference samples."""
        scores = self.score(classifier_attack_score, reconstruction_error)
        self.threshold_ = float(np.quantile(scores, self.threshold_quantile))
        LOGGER.info("Dual consistency threshold fitted at %.6f", self.threshold_)
        return self.threshold_

    def predict(
        self,
        classifier_attack_score: np.ndarray,
        reconstruction_error: np.ndarray,
    ) -> np.ndarray:
        """Predicts binary anomaly labels from hybrid score."""
        if self.threshold_ is None:
            raise RuntimeError("Threshold not fitted. Call fit_threshold first.")
        scores = self.score(classifier_attack_score, reconstruction_error)
        return (scores >= self.threshold_).astype(int)

    def to_dict(self) -> Dict[str, float]:
        """Returns serializable scorer metadata."""
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "threshold_quantile": float(self.threshold_quantile),
            "threshold": float(self.threshold_) if self.threshold_ is not None else None,
        }
