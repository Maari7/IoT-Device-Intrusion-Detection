"""DVC command wrapper for data versioning operations."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Sequence

LOGGER = logging.getLogger(__name__)


class DVCVersioning:
    """Minimal DVC wrapper with explicit command execution and logging."""

    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root)

    def _run(self, args: Sequence[str]) -> subprocess.CompletedProcess:
        """Executes a DVC command and raises on non-zero exit."""
        command: List[str] = ["dvc", *args]
        LOGGER.info("Running command: %s", " ".join(command))
        return subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

    def init(self) -> None:
        """Initializes DVC in the repository."""
        self._run(["init"])

    def remote_add(self, name: str, url: str, default: bool = True) -> None:
        """Adds a DVC remote and optionally sets it as default."""
        self._run(["remote", "add", "-f", name, url])
        if default:
            self._run(["remote", "default", name])

    def add(self, target_path: str | Path) -> None:
        """Adds a target path to DVC tracking."""
        self._run(["add", str(target_path)])

    def repro(self) -> None:
        """Reproduces pipeline stages from dvc.yaml."""
        self._run(["repro"])

    def status(self) -> str:
        """Returns current DVC status output."""
        result = self._run(["status"])
        return result.stdout
