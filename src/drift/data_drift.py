"""Data drift detection using KS test and PSI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureDriftResult:
    """Stores drift metrics for one feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


class DataDriftDetector:
    """Compares reference and current feature distributions."""

    def __init__(self, psi_threshold: float = 0.2, ks_pvalue_threshold: float = 0.05) -> None:
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold

    @staticmethod
    def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Computes population stability index for a numeric feature."""
        ref = reference.replace([np.inf, -np.inf], np.nan).dropna()
        cur = current.replace([np.inf, -np.inf], np.nan).dropna()
        if ref.empty or cur.empty:
            return 0.0

        quantiles = np.linspace(0, 1, bins + 1)
        breakpoints = np.unique(np.quantile(ref, quantiles))
        if len(breakpoints) < 2:
            return 0.0

        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        cur_counts, _ = np.histogram(cur, bins=breakpoints)
        ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts / max(ref_counts.sum(), 1))
        cur_pct = np.where(cur_counts == 0, 0.0001, cur_counts / max(cur_counts.sum(), 1))
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def detect(self, reference: pd.DataFrame, current: pd.DataFrame) -> List[FeatureDriftResult]:
        """Detects drift feature-by-feature."""
        results: List[FeatureDriftResult] = []
        for column in reference.columns:
            if column not in current.columns:
                continue
            ref_series = pd.to_numeric(reference[column], errors="coerce")
            cur_series = pd.to_numeric(current[column], errors="coerce")
            psi_value = self._psi(ref_series, cur_series)
            ks_statistic, ks_pvalue = ks_2samp(ref_series.dropna(), cur_series.dropna())
            drifted = psi_value > self.psi_threshold or ks_pvalue < self.ks_pvalue_threshold
            results.append(
                FeatureDriftResult(
                    feature=column,
                    psi=float(psi_value),
                    ks_statistic=float(ks_statistic),
                    ks_pvalue=float(ks_pvalue),
                    drifted=drifted,
                )
            )
        return results

    def summarize(self, drift_results: List[FeatureDriftResult]) -> Dict[str, float]:
        """Summarizes drift results across features."""
        if not drift_results:
            return {"drifted_feature_count": 0.0, "drifted_feature_ratio": 0.0, "max_psi": 0.0}
        drifted_count = sum(1 for result in drift_results if result.drifted)
        return {
            "drifted_feature_count": float(drifted_count),
            "drifted_feature_ratio": float(drifted_count / len(drift_results)),
            "max_psi": float(max(result.psi for result in drift_results)),
            "mean_psi": float(np.mean([result.psi for result in drift_results])),
        }
