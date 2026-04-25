"""Phase 7 data pipeline wrapper around Phase 1 processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.data.preprocessor import run_phase1
from src.utils.config_loader import load_yaml_config
from src.utils.decorators import timed
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@timed
def run_data_pipeline(config_path: str = "config/config.yaml", max_rows_per_file: int | None = None) -> Dict[str, str]:
    """Executes data ingestion, validation, preprocessing, and split export."""
    config = load_yaml_config(config_path)
    outputs = run_phase1(config=config, max_rows_per_file=max_rows_per_file)
    serialized = {key: str(path) for key, path in outputs.items()}
    LOGGER.info("phase=data_pipeline status=success outputs=%s", serialized)
    return serialized
