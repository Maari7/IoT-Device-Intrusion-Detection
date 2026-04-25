"""Concept drift detection based on performance and confidence degradation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ConceptDriftResult:
    """Stores concept drift signals."""

    f1_drop: float
    auroc_drop: float
    confidence_shift: float
    severity: str


class ConceptDriftDetector:
    """Compares reference and current performance indicators."""

    def __init__(
        self,
        f1_drop_threshold: float = 0.05,
        auroc_drop_threshold: float = 0.05,
        confidence_shift_threshold: float = 0.10,
    ) -> None:
        self.f1_drop_threshold = f1_drop_threshold
        self.auroc_drop_threshold = auroc_drop_threshold
        self.confidence_shift_threshold = confidence_shift_threshold

    def assess(self, reference: Dict[str, float], current: Dict[str, float]) -> ConceptDriftResult:
        """Calculates concept drift severity."""
        f1_drop = max(float(reference.get("f1_weighted", reference.get("f1", 0.0))) - float(current.get("f1_weighted", current.get("f1", 0.0))), 0.0)
        auroc_drop = max(float(reference.get("auroc", 0.0)) - float(current.get("auroc", 0.0)), 0.0)
        confidence_shift = abs(float(reference.get("confidence_mean", 0.0)) - float(current.get("confidence_mean", 0.0)))

        score = 0
        if f1_drop >= self.f1_drop_threshold:
            score += 1
        if auroc_drop >= self.auroc_drop_threshold:
            score += 1
        if confidence_shift >= self.confidence_shift_threshold:
            score += 1

        severity = "low"
        if score >= 3:
            severity = "critical"
        elif score == 2:
            severity = "high"
        elif score == 1:
            severity = "medium"

        return ConceptDriftResult(
            f1_drop=float(f1_drop),
            auroc_drop=float(auroc_drop),
            confidence_shift=float(confidence_shift),
            severity=severity,
        )
