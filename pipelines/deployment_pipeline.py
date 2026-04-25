"""Phase 7 deployment pipeline for model promotion and readiness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.registry.model_manager import ModelManager
from src.utils.config_loader import load_yaml_config
from src.utils.decorators import timed
from src.utils.file_handler import read_json_file, write_json_file
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@timed
def run_deployment_pipeline(
    config_path: str = "config/config.yaml",
    model_config_path: str = "config/model_config.yaml",
    exp1_path: str = "artifacts/research/reports/exp1_results.json",
    exp2_path: str = "artifacts/research/reports/exp2_loao_results.json",
    output_path: str = "artifacts/research/orchestration/deployment_summary.json",
) -> Dict[str, Any]:
    """Runs promotion gate and writes deployment summary."""
    base_config = load_yaml_config(config_path)
    model_config = load_yaml_config(model_config_path)
    merged = {**base_config, **model_config}

    manager = ModelManager(merged)
    decision = manager.promote(exp1_path=exp1_path, exp2_path=exp2_path)
    payload = {
        "approved": bool(decision.approved),
        "reason": decision.reason,
        "champion_source": decision.champion_source,
        "version": decision.version_info.version,
        "metrics": decision.metrics,
    }
    write_json_file(output_path, payload)
    LOGGER.info("phase=deployment_pipeline status=success approved=%s", decision.approved)
    return payload
