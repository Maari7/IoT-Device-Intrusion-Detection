"""Launches the hybrid inference API with uvicorn."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.utils import load_yaml


def parse_args() -> argparse.Namespace:
    """Parses API launch arguments."""
    parser = argparse.ArgumentParser(description="Run the hybrid inference API")
    parser.add_argument("--config", type=str, default="config/inference_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Starts the API server."""
    args = parse_args()
    config = load_yaml(args.config)
    server_cfg = config.get("server", {})
    uvicorn.run(
        "api.main:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=int(server_cfg.get("port", 8000)),
        reload=False,
    )


if __name__ == "__main__":
    main()
