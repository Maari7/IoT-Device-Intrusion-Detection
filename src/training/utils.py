"""Common utilities for Phase 2 training workflows."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configures application logging for training modules."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_yaml(path: str | Path) -> Dict:
    """Loads a YAML file and returns dictionary contents."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_directory(path: str | Path) -> Path:
    """Ensures a directory exists and returns its Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_global_seed(seed: int) -> None:
    """Sets random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


def load_phase1_splits(config: Dict) -> Dict[str, pd.DataFrame]:
    """Loads train/validation/test parquet files produced by Phase 1."""
    paths_cfg = config["paths"]
    output_cfg = config["output"]

    processed_dir = Path(paths_cfg["processed_dir"])
    train_path = processed_dir / output_cfg["train_file"]
    val_path = processed_dir / output_cfg["val_file"]
    test_path = processed_dir / output_cfg["test_file"]

    for required_path in (train_path, val_path, test_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing split file: {required_path}. Run Phase 1 before Phase 2."
            )

    return {
        "train": pd.read_parquet(train_path),
        "val": pd.read_parquet(val_path),
        "test": pd.read_parquet(test_path),
    }


def extract_xy(
    frame: pd.DataFrame,
    target_column: str = "label_id",
    drop_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits features and target from a processed frame."""
    if target_column not in frame.columns:
        raise KeyError(f"Target column '{target_column}' was not found")

    drops = set(drop_columns or ["label", "attack_family", "source_file"])
    drops.add(target_column)
    feature_columns = [c for c in frame.columns if c not in drops]

    x = frame[feature_columns].copy()
    y = frame[target_column].astype(int).copy()
    return x, y


def evaluate_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Computes standard multiclass classification metrics."""
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        classes = np.unique(y_true)
        if len(classes) > 2:
            metrics["auroc_ovr_weighted"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
        elif y_proba.shape[1] >= 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_proba[:, 1]))

    return metrics


def write_json(payload: Dict, output_path: str | Path) -> Path:
    """Writes a dictionary to json file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote JSON artifact to %s", path)
    return path
