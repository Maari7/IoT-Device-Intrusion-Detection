"""Top-level Phase 7 orchestration pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Dict

from pipelines.data_pipeline import run_data_pipeline
from pipelines.deployment_pipeline import run_deployment_pipeline
from pipelines.evaluation_pipeline import run_evaluation_pipeline
from pipelines.training_pipeline import run_training_pipeline
from src.utils.file_handler import write_json_file
from src.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses orchestration pipeline CLI args."""
    parser = argparse.ArgumentParser(description="Run Phase 7 orchestration pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml")
    parser.add_argument("--run-name", type=str, default="phase7_orchestration")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--skip-exp2", action="store_true", help="Skip Experiment 2 LOAO evaluation")
    parser.add_argument("--skip-deployment", action="store_true", help="Skip deployment promotion")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/research/orchestration/phase7_report.json",
    )
    return parser.parse_args()


def run_orchestration(
    config_path: str,
    model_config_path: str,
    run_name: str,
    enable_mlflow: bool,
    run_exp2: bool = True,
    run_deployment: bool = True,
) -> Dict[str, Any]:
    """Runs all pipeline phases in sequence and returns a report."""
    data_result = run_data_pipeline(config_path=config_path)
    training_result = run_training_pipeline(
        config_path=config_path,
        model_config_path=model_config_path,
        run_name=run_name,
        enable_mlflow=enable_mlflow,
    )
    evaluation_result = run_evaluation_pipeline(
        config_path=config_path,
        model_config_path=model_config_path,
        run_exp2=run_exp2,
    )
    deployment_result = (
        run_deployment_pipeline(config_path=config_path, model_config_path=model_config_path)
        if run_deployment
        else {
            "approved": False,
            "reason": "deployment skipped by profile",
            "champion_source": "none",
            "version": "skipped",
            "metrics": {},
            "skipped": True,
        }
    )

    return {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data": data_result,
        "training": {
            "fusion_f1": float(training_result.get("fusion_binary", {}).get("f1", 0.0)),
            "model_version": training_result.get("artifacts", {}).get("fusion_model", "unknown"),
        },
        "evaluation": evaluation_result,
        "deployment": deployment_result,
    }


def main() -> None:
    """CLI entrypoint for full phase-7 orchestration."""
    args = parse_args()
    configure_logging(level="INFO", json_output=True)
    report = run_orchestration(
        config_path=args.config,
        model_config_path=args.model_config,
        run_name=args.run_name,
        enable_mlflow=args.mlflow,
        run_exp2=not args.skip_exp2,
        run_deployment=not args.skip_deployment,
    )
    output_path = write_json_file(args.output, report)
    LOGGER.info("phase=orchestration_pipeline status=success output=%s", output_path)


if __name__ == "__main__":
    main()
