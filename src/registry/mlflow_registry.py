"""MLflow tracking wrapper for experiment logging and artifact management."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator

import mlflow

LOGGER = logging.getLogger(__name__)


class MLflowRegistry:
    """Thin wrapper around MLflow APIs used in Phase 3 experiments."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        LOGGER.info("MLflow configured. URI=%s, experiment=%s", tracking_uri, experiment_name)

    @contextmanager
    def run(self, run_name: str) -> Iterator[str]:
        """Creates a managed MLflow run context."""
        with mlflow.start_run(run_name=run_name) as active_run:
            run_id = active_run.info.run_id
            LOGGER.info("Started MLflow run: %s", run_id)
            yield run_id
            LOGGER.info("Finished MLflow run: %s", run_id)

    @staticmethod
    def log_params(params: Dict) -> None:
        """Logs parameter dictionary to active MLflow run."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    @staticmethod
    def log_metrics(metrics: Dict[str, float], step: int | None = None, prefix: str = "") -> None:
        """Logs metrics dictionary to active MLflow run."""
        for key, value in metrics.items():
            metric_key = f"{prefix}{key}" if prefix else key
            if step is None:
                mlflow.log_metric(metric_key, float(value))
            else:
                mlflow.log_metric(metric_key, float(value), step=step)

    @staticmethod
    def log_artifacts(paths: Dict[str, str]) -> None:
        """Logs artifact files to active MLflow run."""
        for artifact_name, artifact_path in paths.items():
            path = Path(artifact_path)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=artifact_name)
