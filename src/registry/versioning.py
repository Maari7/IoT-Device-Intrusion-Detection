"""Versioning helpers for research model promotion artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class ModelVersionInfo:
    """Serializable metadata for a promoted model version."""

    version: str
    model_name: str
    data_hash: str
    config_hash: str
    experiment_hash: str
    promoted_from: str

    def to_dict(self) -> Dict[str, str]:
        """Returns a plain dictionary representation."""
        return asdict(self)


class VersionManager:
    """Creates reproducible semantic-like version identifiers."""

    def __init__(self, namespace: str = "hybrid") -> None:
        self.namespace = namespace

    @staticmethod
    def _hash_bytes(payload: bytes) -> str:
        """Returns SHA256 hash hex digest for bytes payload."""
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def hash_file(path: str | Path) -> str:
        """Computes SHA256 hash for a file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot hash missing file: {file_path}")
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    @staticmethod
    def hash_json(payload: Dict[str, Any]) -> str:
        """Computes SHA256 hash for a JSON-serializable dictionary."""
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def build_version(self, components: Iterable[str]) -> str:
        """Builds a compact version identifier from hash components."""
        digest = self._hash_bytes("|".join(list(components)).encode("utf-8"))
        return f"{self.namespace}-{digest[:12]}"
