"""Experiment 1: compare baseline XGBoost vs hybrid AGA+fusion pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hybrid_trainer import HybridTrainingOrchestrator


def parse_args() -> argparse.Namespace:
    """Parses experiment command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Experiment 1: AE vs baseline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/research/reports/exp1_results.json",
        help="Where to write experiment result summary",
    )
    return parser.parse_args()


def main() -> None:
    """Runs experiment and writes result JSON."""
    args = parse_args()
    orchestrator = HybridTrainingOrchestrator(
        config_path=args.config,
        model_config_path=args.model_config,
    )

    results = orchestrator.run(enable_mlflow=args.mlflow, run_name="exp1_ae_vs_baseline")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
