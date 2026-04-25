"""CLI for Phase 4 model promotion and registry gating."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.model_manager import ModelManager
from src.training.utils import load_yaml


def parse_args() -> argparse.Namespace:
    """Parses CLI options for promotion."""
    parser = argparse.ArgumentParser(description="Run Phase 4 promotion gate")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml")
    parser.add_argument(
        "--exp1",
        type=str,
        default="artifacts/research/reports/exp1_results.json",
        help="Path to Exp1 result JSON",
    )
    parser.add_argument(
        "--exp2",
        type=str,
        default="artifacts/research/reports/exp2_loao_results.json",
        help="Path to Exp2 result JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/research/registry/promotion_decision.json",
        help="Where to write the promotion decision",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the registry promotion evaluation and writes decision JSON."""
    args = parse_args()
    config = load_yaml(args.config)
    model_config = load_yaml(args.model_config)
    merged_config = {**config, **model_config}

    manager = ModelManager(merged_config)
    decision = manager.promote(exp1_path=args.exp1, exp2_path=args.exp2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(decision.to_dict(), indent=2))


if __name__ == "__main__":
    main()
