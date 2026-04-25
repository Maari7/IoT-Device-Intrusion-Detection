"""SHAP-based feature attribution weighting for hybrid IDS pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class ShapWeightGenerator:
    """Builds SHAP-derived feature weights using a tree surrogate model."""

    def __init__(self, base_model, random_state: int = 42, sample_rows: int = 1000) -> None:
        self.base_model = base_model
        self.random_state = random_state
        self.sample_rows = int(sample_rows)
        self.feature_names: List[str] = []
        self.feature_weights: pd.Series | None = None



    def fit(self, x_train: pd.DataFrame, y_train: pd.Series = None) -> "ShapWeightGenerator":
        """Uses TreeSHAP on the already-trained base model (NO surrogate)."""
        if not hasattr(self, "base_model") or self.base_model is None:
            raise ValueError("base_model must be provided for TreeSHAP")

        self.feature_names = list(x_train.columns)

        try:
            import shap
        except ImportError:
            LOGGER.warning(
                "shap is not installed; falling back to XGBoost feature_importances_ for weighting"
            )
            raw_weights = pd.Series(
                getattr(self.base_model, "feature_importances_", np.ones(len(self.feature_names))),
                index=self.feature_names,
                dtype="float64",
            )
            if float(raw_weights.max()) == float(raw_weights.min()):
                normalized = pd.Series(np.ones(len(raw_weights)), index=self.feature_names, dtype="float64")
            else:
                normalized = (raw_weights - raw_weights.min()) / (raw_weights.max() - raw_weights.min())
                normalized = normalized.clip(lower=0.05)

            self.feature_weights = normalized
            LOGGER.info("Computed fallback feature weights for %d features", len(self.feature_weights))
            return self

        explainer = shap.TreeExplainer(self.base_model)
        x_sample = x_train.sample(n=min(self.sample_rows, len(x_train)), random_state=self.random_state)
        shap_values = explainer.shap_values(x_sample)

        if isinstance(shap_values, list):
            stacked = np.stack([np.abs(v) for v in shap_values], axis=0)
            mean_abs = stacked.mean(axis=(0, 1))
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(arr).mean(axis=0)

        raw_weights = pd.Series(mean_abs, index=self.feature_names, dtype="float64")
        if float(raw_weights.max()) == float(raw_weights.min()):
            normalized = pd.Series(np.ones(len(raw_weights)), index=self.feature_names, dtype="float64")
        else:
            normalized = (raw_weights - raw_weights.min()) / (raw_weights.max() - raw_weights.min())
            normalized = normalized.clip(lower=0.05)

        self.feature_weights = normalized
        LOGGER.info("Computed SHAP weights (TreeSHAP on base model, no surrogate)")
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Applies feature weighting to an input frame."""
        if self.feature_weights is None:
            raise RuntimeError("ShapWeightGenerator must be fitted before transform")

        missing = [col for col in self.feature_weights.index if col not in x.columns]
        if missing:
            raise ValueError(f"Input is missing weighted columns: {missing[:5]}")

        weighted = x[self.feature_weights.index].mul(self.feature_weights, axis=1)
        return weighted

    def fit_transform(self, x_train: pd.DataFrame, y_train: pd.Series = None) -> pd.DataFrame:
        """Fits weighting generator and transforms train data."""
        self.fit(x_train)
        return self.transform(x_train)

    def save(self, output_path: str | Path) -> Path:
        """Persists learned feature weights to JSON."""
        if self.feature_weights is None:
            raise RuntimeError("No weights to save")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_weights": {k: float(v) for k, v in self.feature_weights.to_dict().items()},
            "feature_names": self.feature_names,
            "random_state": self.random_state,
            "sample_rows": self.sample_rows,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved SHAP feature weights to %s", path)
        return path

    @staticmethod
    def load(input_path: str | Path) -> "ShapWeightGenerator":
        """Loads saved SHAP weights from JSON."""
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        instance = ShapWeightGenerator(
            base_model=None,
            random_state=int(payload.get("random_state", 42)),
            sample_rows=int(payload.get("sample_rows", 1000)),
        )
        feature_weights = payload["feature_weights"]
        instance.feature_weights = pd.Series(feature_weights, dtype="float64")
        instance.feature_names = list(payload.get("feature_names", instance.feature_weights.index.tolist()))
        return instance
