"""Model loading utilities for the hybrid inference service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from src.research.autoencoder import AttributionGuidedAutoencoder
from src.research.dual_consistency import DualConsistencyScorer
from src.research.fusion import FusionModel
from src.research.shap_weighting import ShapWeightGenerator

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceArtifacts:
    """Holds all loaded inference-time artifacts."""

    feature_columns: List[str]
    preprocessor: Dict[str, Any]
    baseline_model: Any
    fusion_model: Any
    autoencoder: AttributionGuidedAutoencoder
    shap_weights: ShapWeightGenerator
    dual_consistency: DualConsistencyScorer
    model_version: str
    champion_source: str
    artifact_dir: Path


class InferenceModelLoader:
    """Loads preprocessing and hybrid model artifacts from disk."""

    def __init__(self, inference_config: Dict[str, Any], promotion_manifest_path: str | Path | None = None) -> None:
        self.inference_config = inference_config
        self.artifact_cfg = inference_config.get("artifacts", {})
        self.prediction_cfg = inference_config.get("prediction", {})
        self.promotion_manifest_path = Path(
            promotion_manifest_path or "artifacts/research/registry/promotion_decision.json"
        )

    def _load_preprocessor(self, path: Path) -> Dict[str, Any]:
        """Loads the Phase 1 preprocessor payload."""
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found: {path}")
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise ValueError("Invalid preprocessor payload")
        return payload

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads the promotion decision manifest if available."""
        if not self.promotion_manifest_path.exists():
            return {}
        return json.loads(self.promotion_manifest_path.read_text(encoding="utf-8"))

    def _load_baseline_model(self, artifact_dir: Path, file_name: str) -> Any:
        """Loads the baseline XGBoost model artifact."""
        path = artifact_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Baseline model not found: {path}")
        return joblib.load(path)

    def _load_fusion_model(self, artifact_dir: Path, file_name: str) -> FusionModel:
        """Loads the fusion classifier artifact."""
        path = artifact_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Fusion model not found: {path}")
        model = joblib.load(path)
        fusion = FusionModel(config={})
        fusion.model = model
        return fusion

    def _load_autoencoder(self, artifact_dir: Path, file_name: str) -> AttributionGuidedAutoencoder:
        """Loads autoencoder weights and configuration."""
        path = artifact_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Autoencoder artifact not found: {path}")

        payload = torch.load(path, map_location="cpu")
        autoencoder = AttributionGuidedAutoencoder(payload.get("config", {}))
        feature_names = list(payload.get("feature_names", []))
        feature_weights = np.asarray(payload.get("feature_weights", []), dtype=np.float32)
        model_state = payload.get("state_dict")

        if not feature_names or model_state is None:
            raise ValueError("Invalid autoencoder artifact payload")

        autoencoder.feature_names = feature_names
        autoencoder.feature_weights = feature_weights
        autoencoder.model = None

        input_dim = len(feature_names)
        hidden_dims = payload.get("config", {}).get("hidden_dims", [128, 64])
        latent_dim = int(payload.get("config", {}).get("latent_dim", 32))
        model = autoencoder._AutoEncoderNet(  # type: ignore[attr-defined]
            input_dim=input_dim,
            hidden_dims=list(hidden_dims),
            latent_dim=latent_dim,
        )
        model.load_state_dict(model_state)
        model.eval()
        autoencoder.model = model
        return autoencoder

    def _load_shap_weights(self, artifact_dir: Path, file_name: str) -> ShapWeightGenerator:
        """Loads SHAP weight JSON payload."""
        path = artifact_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"SHAP weights not found: {path}")
        return ShapWeightGenerator.load(path)

    def _load_dual_consistency(self, artifact_dir: Path, file_name: str) -> DualConsistencyScorer:
        """Loads dual consistency configuration."""
        path = artifact_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Dual consistency artifact not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        scorer = DualConsistencyScorer(
            alpha=float(payload.get("alpha", 0.5)),
            beta=float(payload.get("beta", 0.5)),
            threshold_quantile=float(payload.get("threshold_quantile", 0.95)),
        )
        scorer.threshold_ = payload.get("threshold")
        return scorer

    def load(self) -> InferenceArtifacts:
        """Loads all inference artifacts and returns a bundle."""
        preprocessor_path = Path(self.artifact_cfg.get("preprocessor_path", "data/features/preprocessor.joblib"))
        preprocessor_payload = self._load_preprocessor(preprocessor_path)
        feature_columns = list(preprocessor_payload.get("feature_columns", []))
        if not feature_columns:
            raise ValueError("Preprocessor payload does not contain feature columns")

        manifest = self._load_manifest()
        artifact_dir = Path(
            manifest.get("artifacts", {}).get("baseline_model", self.artifact_cfg.get("model_dir", "artifacts/research/models"))
        )
        artifact_dir = Path(self.artifact_cfg.get("model_dir", artifact_dir))

        baseline_model = self._load_baseline_model(artifact_dir, self.artifact_cfg.get("baseline_model", "xgboost_baseline.joblib"))
        fusion_model = self._load_fusion_model(artifact_dir, self.artifact_cfg.get("fusion_model", "fusion_model.joblib"))
        autoencoder = self._load_autoencoder(artifact_dir, self.artifact_cfg.get("autoencoder_model", "aga_autoencoder.pt"))
        shap_weights = self._load_shap_weights(artifact_dir, self.artifact_cfg.get("shap_weights", "shap_feature_weights.json"))
        dual_consistency = self._load_dual_consistency(artifact_dir, self.artifact_cfg.get("dual_consistency", "dual_consistency.json"))

        model_version = manifest.get("version_info", {}).get("version", "unversioned")
        champion_source = manifest.get("champion_source", "unknown")

        return InferenceArtifacts(
            feature_columns=feature_columns,
            preprocessor=preprocessor_payload,
            baseline_model=baseline_model,
            fusion_model=fusion_model,
            autoencoder=autoencoder,
            shap_weights=shap_weights,
            dual_consistency=dual_consistency,
            model_version=model_version,
            champion_source=champion_source,
            artifact_dir=artifact_dir,
        )
