"""File IO helpers used by pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    """Ensures that a directory exists."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json_file(path: str | Path) -> Dict[str, Any]:
    """Reads a JSON file and returns a dictionary."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload in {file_path} must be a dictionary")
    return payload


def write_json_file(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Writes a dictionary to JSON file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return file_path
