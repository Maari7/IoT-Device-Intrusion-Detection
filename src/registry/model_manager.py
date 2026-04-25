"""Model promotion and registry gating for research experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.registry.mlflow_registry import MLflowRegistry
from src.registry.versioning import ModelVersionInfo, VersionManager

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionDecision:
    """Holds promotion decision details for a candidate model."""

    approved: bool
    champion_source: str
    reason: str
    version_info: ModelVersionInfo
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the decision payload."""
        return {
            "approved": self.approved,
            "champion_source": self.champion_source,
            "reason": self.reason,
            "version_info": self.version_info.to_dict(),
            "metrics": self.metrics,
        }


class ModelManager:
    """Evaluates experiment outputs and promotes a champion if gates pass."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        registry_cfg = config.get("registry", {})
        self.thresholds = registry_cfg.get("thresholds", {})
        self.output_dir = Path(registry_cfg.get("output_dir", "artifacts/research/registry"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.version_manager = VersionManager(namespace=registry_cfg.get("version_namespace", "hybrid"))

    @staticmethod
    def _load_json(path: str | Path) -> Dict[str, Any]:
        """Loads JSON artifact from disk."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing artifact: {file_path}")
        return json.loads(file_path.read_text(encoding="utf-8"))

    def _extract_exp1_metrics(self, exp1_result: Dict[str, Any]) -> Dict[str, float]:
        """Pulls the primary metrics used for promotion from Exp1."""
        baseline = exp1_result.get("baseline_multiclass", {})
        hybrid = exp1_result.get("hybrid_binary", {})
        fusion = exp1_result.get("fusion_binary", {})
        return {
            "baseline_f1_weighted": float(baseline.get("f1_weighted", 0.0)),
            "hybrid_f1": float(hybrid.get("f1", 0.0)),
            "fusion_f1": float(fusion.get("f1", 0.0)),
            "fusion_auroc": float(fusion.get("auroc", fusion.get("auroc_ovr_weighted", 0.0))),
        }

    def _extract_loao_metrics(self, exp2_result: Dict[str, Any]) -> Dict[str, float]:
        """Aggregates LOAO robustness metrics across held-out attacks."""
        runs = exp2_result.get("runs", [])
        if not runs:
            return {"loao_mean_f1": 0.0, "loao_min_f1": 0.0, "loao_run_count": 0.0}

        f1_scores: List[float] = []
        attack_rates: List[float] = []
        for run in runs:
            fusion_metrics = run.get("fusion_binary", {})
            f1_scores.append(float(fusion_metrics.get("f1", 0.0)))
            attack_rates.append(float(fusion_metrics.get("attack_rate_pred", 0.0)))

        return {
            "loao_mean_f1": float(pd.Series(f1_scores).mean()),
            "loao_min_f1": float(pd.Series(f1_scores).min()),
            "loao_mean_attack_rate_pred": float(pd.Series(attack_rates).mean()),
            "loao_run_count": float(len(runs)),
        }

    def _passes_gates(self, metrics: Dict[str, float]) -> tuple[bool, str]:
        """Checks whether the metrics satisfy promotion thresholds."""
        min_fusion_f1 = float(self.thresholds.get("min_fusion_f1", 0.0))
        min_loao_mean_f1 = float(self.thresholds.get("min_loao_mean_f1", 0.0))
        min_loao_min_f1 = float(self.thresholds.get("min_loao_min_f1", 0.0))
        min_baseline_gap = float(self.thresholds.get("min_fusion_over_baseline_gap", 0.0))
        min_fusion_auroc = float(self.thresholds.get("min_fusion_auroc", 0.0))

        if metrics["fusion_f1"] < min_fusion_f1:
            return False, f"fusion_f1 below threshold ({metrics['fusion_f1']:.4f} < {min_fusion_f1:.4f})"
        if metrics["fusion_auroc"] < min_fusion_auroc:
            return False, f"fusion_auroc below threshold ({metrics['fusion_auroc']:.4f} < {min_fusion_auroc:.4f})"
        if metrics["loao_mean_f1"] < min_loao_mean_f1:
            return False, f"loao_mean_f1 below threshold ({metrics['loao_mean_f1']:.4f} < {min_loao_mean_f1:.4f})"
        if metrics["loao_min_f1"] < min_loao_min_f1:
            return False, f"loao_min_f1 below threshold ({metrics['loao_min_f1']:.4f} < {min_loao_min_f1:.4f})"
        if (metrics["fusion_f1"] - metrics["baseline_f1_weighted"]) < min_baseline_gap:
            return False, (
                "fusion_f1 does not improve enough over baseline "
                f"({metrics['fusion_f1']:.4f} - {metrics['baseline_f1_weighted']:.4f} < {min_baseline_gap:.4f})"
            )
        return True, "passed promotion gates"

    def promote(self, exp1_path: str | Path, exp2_path: str | Path) -> PromotionDecision:
        """Evaluates experiment results, writes a manifest, and returns the decision."""
        exp1_result = self._load_json(exp1_path)
        exp2_result = self._load_json(exp2_path)

        exp1_metrics = self._extract_exp1_metrics(exp1_result)
        loao_metrics = self._extract_loao_metrics(exp2_result)
        combined_metrics = {**exp1_metrics, **loao_metrics}

        approved, reason = self._passes_gates(combined_metrics)

        version_info = ModelVersionInfo(
            version=self.version_manager.build_version(
                [
                    self.version_manager.hash_json(exp1_result),
                    self.version_manager.hash_json(exp2_result),
                    self.version_manager.hash_json(self.thresholds),
                ]
            ),
            model_name="fusion",
            data_hash=self.version_manager.hash_json({"exp1": exp1_result.get("run_name"), "exp2": exp2_result.get("total_runs", 0)}),
            config_hash=self.version_manager.hash_json(self.config),
            experiment_hash=self.version_manager.hash_json({"exp1": exp1_result, "exp2": exp2_result}),
            promoted_from="exp1_ae_vs_baseline + exp2_loao",
        )

        decision = PromotionDecision(
            approved=approved,
            champion_source="fusion" if approved else "none",
            reason=reason,
            version_info=version_info,
            metrics=combined_metrics,
        )

        manifest_path = self.output_dir / f"{version_info.version}_manifest.json"
        manifest_path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
        LOGGER.info("Wrote promotion manifest to %s", manifest_path)

        if approved:
            mlflow_cfg = self.config.get("mlflow", {})
            registry = MLflowRegistry(
                tracking_uri=mlflow_cfg.get("tracking_uri", "file:./mlruns"),
                experiment_name=mlflow_cfg.get("experiment_name", "hx-idl-hybrid"),
            )
            with registry.run(run_name=f"promote_{version_info.version}"):
                registry.log_params(
                    {
                        "decision": "approved",
                        "version": version_info.version,
                        "model_name": version_info.model_name,
                    }
                )
                registry.log_metrics(combined_metrics, prefix="promotion_")
                registry.log_artifacts(
                    {
                        "exp1": str(Path(exp1_path)),
                        "exp2": str(Path(exp2_path)),
                        "manifest": str(manifest_path),
                    }
                )

        return decision
