"""Batch scoring utilities for the hybrid inference service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.inference.predictor import HybridPredictor, PredictionResult


@dataclass
class BatchScoreSummary:
    """Aggregated summary for a scored batch."""

    total_items: int
    attack_count: int
    benign_count: int
    mean_confidence: float


class BatchScorer:
    """Scores batches and returns both item-level and aggregate outputs."""

    def __init__(self, predictor: HybridPredictor) -> None:
        self.predictor = predictor

    def score(self, items: List[Dict[str, float]]) -> tuple[List[PredictionResult], BatchScoreSummary]:
        """Scores a batch of items and summarizes the results."""
        results = self.predictor.predict_batch(items)
        attack_count = sum(1 for result in results if result.prediction == 1)
        benign_count = len(results) - attack_count
        mean_confidence = sum(result.confidence for result in results) / max(len(results), 1)
        summary = BatchScoreSummary(
            total_items=len(results),
            attack_count=attack_count,
            benign_count=benign_count,
            mean_confidence=mean_confidence,
        )
        return results, summary
