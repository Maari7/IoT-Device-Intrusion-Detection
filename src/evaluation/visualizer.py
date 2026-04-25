"""Visualization utilities for hybrid experiment analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, confusion_matrix


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[str],
    output_path: str | Path,
    title: str,
) -> Path:
    """Saves confusion matrix heatmap plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(list(labels))))
    ax.set_yticks(np.arange(len(list(labels))))
    ax.set_xticklabels(list(labels), rotation=45, ha="right")
    ax.set_yticklabels(list(labels))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_binary_roc(y_true: np.ndarray, y_score: np.ndarray, output_path: str | Path, title: str) -> Path:
    """Saves ROC curve for binary anomaly detection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_score, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
