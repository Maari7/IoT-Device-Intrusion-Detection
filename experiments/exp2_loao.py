"""Experiment 2: Leave-One-Attack-Out evaluation for fusion robustness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.loao import LeaveOneAttackOut
from src.training.hybrid_trainer import HybridTrainingOrchestrator
from src.training.utils import load_phase1_splits, load_yaml


def parse_args() -> argparse.Namespace:
    """Parses LOAO experiment arguments."""
    parser = argparse.ArgumentParser(description="Run Experiment 2: LOAO")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/research/reports/exp2_loao_results.json",
        help="Where to write LOAO summary",
    )
    return parser.parse_args()


def main() -> None:
    """Runs LOAO experiments across all known attack labels."""
    args = parse_args()
    config = load_yaml(args.config)
    splits = load_phase1_splits(config)
    merged = pd.concat([splits["train"], splits["val"], splits["test"]], axis=0, ignore_index=True)

    # Extract ALL possible labels upfront to avoid "unseen label" errors during LOAO
    all_unique_labels = sorted(merged["label"].astype(str).unique().tolist())

    loao = LeaveOneAttackOut(label_column="label", benign_label="benign")
    trainer = HybridTrainingOrchestrator(config_path=args.config, model_config_path=args.model_config)

    runs = []
    for split in loao.split(merged):
        run_name = f"exp2_loao_{split.held_out_attack}"
        result = trainer.run_on_frames(
            train_df=split.train_df,
            test_df=split.test_df,
            run_name=run_name,
            all_labels=all_unique_labels  # Pass full label set to encoder
        )
        result["held_out_attack"] = split.held_out_attack
        runs.append(result)

    summary = {
        "total_runs": len(runs),
        "held_out_attacks": [r["held_out_attack"] for r in runs],
        "runs": runs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output_path), "total_runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
