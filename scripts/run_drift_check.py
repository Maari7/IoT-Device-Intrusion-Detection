"""Runs data drift, concept drift, and retraining evaluation for Phase 6."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.drift_analyzer import DriftAnalyzer
from src.drift.retraining_trigger import RetrainingTrigger
from src.training.utils import load_yaml


def parse_args() -> argparse.Namespace:
    """Parses drift check CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Phase 6 drift monitoring")
    parser.add_argument("--config", type=str, default="config/drift_config.yaml")
    parser.add_argument(
        "--reference-data",
        type=str,
        default="data/processed/train.parquet",
        help="Reference feature dataset",
    )
    parser.add_argument(
        "--current-data",
        type=str,
        default="data/processed/test.parquet",
        help="Current feature dataset to compare",
    )
    parser.add_argument(
        "--reference-metrics",
        type=str,
        default="artifacts/research/reports/exp1_results.json",
        help="Reference metrics JSON",
    )
    parser.add_argument(
        "--current-metrics",
        type=str,
        default="artifacts/research/reports/exp2_loao_results.json",
        help="Current metrics JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/research/monitoring/drift_report.json",
        help="Where to write drift report",
    )
    parser.add_argument(
        "--retrain-output",
        type=str,
        default="artifacts/research/monitoring/retraining_decision.json",
        help="Where to write retraining decision",
    )
    return parser.parse_args()


def _load_json(path: str | Path) -> Dict:
    """Loads a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_metrics(payload: Dict) -> Dict[str, float]:
    """Extracts a compact metric dict from experiment outputs."""
    if "fusion_binary" in payload:
        fusion = payload.get("fusion_binary", {})
        return {
            "f1": float(fusion.get("f1", 0.0)),
            "auroc": float(fusion.get("auroc", fusion.get("auroc_ovr_weighted", 0.0))),
            "confidence_mean": float(fusion.get("confidence_mean", fusion.get("attack_rate_pred", 0.0))),
        }
    if "hybrid_binary" in payload:
        hybrid = payload.get("hybrid_binary", {})
        fusion = payload.get("fusion_binary", {})
        return {
            "f1": float(fusion.get("f1", hybrid.get("f1", 0.0))),
            "auroc": float(fusion.get("auroc", 0.0)),
            "confidence_mean": float(fusion.get("confidence_mean", 0.0)),
        }
    return {
        "f1": float(payload.get("f1", 0.0)),
        "auroc": float(payload.get("auroc", 0.0)),
        "confidence_mean": float(payload.get("confidence_mean", 0.0)),
    }


def _strip_non_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Keeps only numeric feature columns for drift comparison."""
    drops = {"label", "attack_family", "source_file", "label_id"}
    feature_columns = [column for column in frame.columns if column not in drops]
    return frame[feature_columns].copy()


def main() -> None:
    """Runs drift analysis and retraining decision writing."""
    args = parse_args()
    drift_config = load_yaml(args.config)
    analyzer = DriftAnalyzer(drift_config)
    retraining_trigger = RetrainingTrigger(drift_config)

    reference_df = pd.read_parquet(args.reference_data)
    current_df = pd.read_parquet(args.current_data)

    reference_features = _strip_non_features(reference_df)
    current_features = _strip_non_features(current_df)

    reference_metrics = _extract_metrics(_load_json(args.reference_metrics))
    current_metrics = _extract_metrics(_load_json(args.current_metrics))

    report = analyzer.analyze(
        reference_features=reference_features,
        current_features=current_features,
        reference_metrics=reference_metrics,
        current_metrics=current_metrics,
    )
    report_path = analyzer.save(report, args.output)

    decision = retraining_trigger.evaluate(json.loads(report_path.read_text(encoding="utf-8")))
    decision_path = retraining_trigger.save(decision, args.retrain_output)

    print(
        json.dumps(
            {
                "status": "ok",
                "drift_report": str(report_path),
                "retraining_decision": str(decision_path),
                "severity": report.severity,
                "trigger": decision.trigger,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
