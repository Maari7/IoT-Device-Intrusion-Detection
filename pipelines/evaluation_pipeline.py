"""Phase 7 evaluation pipeline for Exp1 and Exp2 experiments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from src.utils.decorators import timed
from src.utils.file_handler import read_json_file
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@timed
def run_evaluation_pipeline(
    config_path: str = "config/config.yaml",
    model_config_path: str = "config/model_config.yaml",
    exp1_phase3_path: str = "artifacts/research/reports/phase3_results.json",
    exp1_result_path: str = "artifacts/research/reports/exp1_results.json",
    exp2_result_path: str = "artifacts/research/reports/exp2_loao_results.json",
    run_exp2: bool = True,
) -> Dict[str, Any]:
    """Runs experiment scripts and returns compact evaluation summary."""
    exp1_path = Path(exp1_result_path)
    if not exp1_path.exists():
        fallback_path = Path(exp1_phase3_path)
        if fallback_path.exists():
            exp1_path.parent.mkdir(parents=True, exist_ok=True)
            exp1_path.write_text(fallback_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            subprocess.run(
                [
                    sys.executable,
                    "experiments/exp1_ae_vs_baseline.py",
                    "--config",
                    config_path,
                    "--model-config",
                    model_config_path,
                ],
                check=True,
            )
    if run_exp2:
        subprocess.run(
            [
                sys.executable,
                "experiments/exp2_loao.py",
                "--config",
                config_path,
                "--model-config",
                model_config_path,
            ],
            check=True,
        )

    exp1 = read_json_file(exp1_result_path)
    exp2 = read_json_file(exp2_result_path) if run_exp2 else {}

    summary = {
        "exp1_run_name": exp1.get("run_name", "unknown"),
        "exp1_fusion_f1": float(exp1.get("fusion_binary", {}).get("f1", 0.0)),
        "exp2_enabled": bool(run_exp2),
        "exp2_total_runs": int(exp2.get("total_runs", 0)) if run_exp2 else 0,
        "exp2_attacks": exp2.get("held_out_attacks", []) if run_exp2 else [],
    }
    LOGGER.info("phase=evaluation_pipeline status=success summary=%s", summary)
    return summary
