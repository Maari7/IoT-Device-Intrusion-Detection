"""CLI entrypoint for hybrid research training and Phase 3 tracking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hybrid_trainer import HybridTrainingOrchestrator

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line options for hybrid training."""
    parser = argparse.ArgumentParser(description="Run hybrid IDS training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to primary project config",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument(
        "--run-name",
        type=str,
        default="phase3_hybrid_train",
        help="Experiment run name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/research/reports/train_command_result.json",
        help="Path to save command-level output summary",
    )
    return parser.parse_args()


def main() -> None:
    """Runs hybrid training and writes a summary artifact."""
    args = parse_args()

    orchestrator = HybridTrainingOrchestrator(
        config_path=args.config,
        model_config_path=args.model_config,
    )

    results = orchestrator.run(enable_mlflow=args.mlflow, run_name=args.run_name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "status": "ok",
            "run_name": results.get("run_name"),
            "baseline_f1_weighted": results.get("baseline_multiclass", {}).get("f1_weighted"),
            "fusion_f1": results.get("fusion_binary", {}).get("f1"),
            "summary": str(output_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
