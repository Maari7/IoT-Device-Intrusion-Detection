"""Evaluation orchestrator for hybrid IDS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.evaluation.metrics import binary_metrics, multiclass_metrics


@dataclass(frozen=True)
class HybridEvaluationResult:
    """Container for baseline and hybrid model metric groups."""

    baseline_multiclass: Dict[str, float]
    hybrid_binary: Dict[str, float]
    fusion_binary: Dict[str, float]


class HybridEvaluator:
    """Computes standardized metrics for hybrid experiment outputs."""

    def evaluate(
        self,
        y_true_multiclass: np.ndarray,
        y_pred_multiclass: np.ndarray,
        y_proba_multiclass: np.ndarray,
        y_true_binary: np.ndarray,
        y_pred_hybrid_binary: np.ndarray,
        y_score_hybrid_binary: np.ndarray,
        y_pred_fusion_binary: np.ndarray,
        y_score_fusion_binary: np.ndarray,
    ) -> HybridEvaluationResult:
        """Evaluates baseline multiclass and hybrid/fusion binary outputs."""
        baseline = multiclass_metrics(
            y_true=y_true_multiclass,
            y_pred=y_pred_multiclass,
            y_proba=y_proba_multiclass,
        )
        hybrid = binary_metrics(
            y_true=y_true_binary,
            y_pred=y_pred_hybrid_binary,
            y_score=y_score_hybrid_binary,
        )
        fusion = binary_metrics(
            y_true=y_true_binary,
            y_pred=y_pred_fusion_binary,
            y_score=y_score_fusion_binary,
        )

        return HybridEvaluationResult(
            baseline_multiclass=baseline,
            hybrid_binary=hybrid,
            fusion_binary=fusion,
        )
