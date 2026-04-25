"""Phase 7 training pipeline wrapper for hybrid model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.training.hybrid_trainer import HybridTrainingOrchestrator
from src.utils.decorators import timed
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@timed
def run_training_pipeline(
    config_path: str = "config/config.yaml",
    model_config_path: str = "config/model_config.yaml",
    run_name: str = "phase7_training",
    enable_mlflow: bool = False,
) -> Dict[str, Any]:
    """Executes hybrid training and returns metrics/artifact summary."""
    orchestrator = HybridTrainingOrchestrator(config_path=config_path, model_config_path=model_config_path)
    results = orchestrator.run(enable_mlflow=enable_mlflow, run_name=run_name)
    LOGGER.info("phase=training_pipeline status=success run_name=%s", run_name)
    return results
