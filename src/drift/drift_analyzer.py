"""Combines data and concept drift into a single operational report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.drift.concept_drift import ConceptDriftDetector, ConceptDriftResult
from src.drift.data_drift import DataDriftDetector, FeatureDriftResult


@dataclass(frozen=True)
class DriftReport:
    """Full drift report across data and concept dimensions."""

    data_drift: Dict[str, float]
    concept_drift: Dict[str, float]
    severity: str
    drifted_features: List[Dict]


class DriftAnalyzer:
    """Analyzes both data and concept drift and prepares a report."""

    def __init__(self, drift_config: Dict) -> None:
        data_cfg = drift_config.get("drift", {}).get("data", {})
        concept_cfg = drift_config.get("drift", {}).get("concept", {})
        self.data_detector = DataDriftDetector(
            psi_threshold=float(data_cfg.get("psi_threshold", 0.2)),
            ks_pvalue_threshold=float(data_cfg.get("ks_pvalue_threshold", 0.05)),
        )
        self.concept_detector = ConceptDriftDetector(
            f1_drop_threshold=float(concept_cfg.get("f1_drop_threshold", 0.05)),
            auroc_drop_threshold=float(concept_cfg.get("auroc_drop_threshold", 0.05)),
            confidence_shift_threshold=float(concept_cfg.get("confidence_shift_threshold", 0.10)),
        )

    def analyze(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
    ) -> DriftReport:
        """Returns a drift report object."""
        feature_results = self.data_detector.detect(reference_features, current_features)
        data_summary = self.data_detector.summarize(feature_results)
        concept_result = self.concept_detector.assess(reference_metrics, current_metrics)
        severity = "low"
        if data_summary.get("drifted_feature_ratio", 0.0) >= 0.5 or concept_result.severity == "critical":
            severity = "critical"
        elif data_summary.get("drifted_feature_ratio", 0.0) >= 0.25 or concept_result.severity == "high":
            severity = "high"
        elif data_summary.get("drifted_feature_ratio", 0.0) > 0.0 or concept_result.severity in {"medium", "high", "critical"}:
            severity = "medium"

        return DriftReport(
            data_drift=data_summary,
            concept_drift=asdict(concept_result),
            severity=severity,
            drifted_features=[asdict(result) for result in feature_results if result.drifted],
        )

    @staticmethod
    def save(report: DriftReport, output_path: str | Path) -> Path:
        """Saves a drift report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        return path
