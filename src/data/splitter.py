"""Train/validation/test splitting utilities for labeled data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic stratified splitting."""

    test_size: float
    val_size: float
    random_state: int
    stratify_column: str


class DatasetSplitter:
    """Performs deterministic stratified train/val/test splits."""

    def __init__(self, split_config: SplitConfig) -> None:
        if not 0 < split_config.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < split_config.val_size < 1:
            raise ValueError("val_size must be between 0 and 1")
        if split_config.test_size + split_config.val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1")

        self.cfg = split_config

    def split(self, frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Splits a labeled DataFrame into train/val/test datasets."""
        stratify_col = self.cfg.stratify_column
        if stratify_col not in frame.columns:
            raise KeyError(f"Stratification column '{stratify_col}' not found in dataframe")

        y = frame[stratify_col]

        train_val_df, test_df = train_test_split(
            frame,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            shuffle=True,
            stratify=y,
        )

        adjusted_val_size = self.cfg.val_size / (1.0 - self.cfg.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            random_state=self.cfg.random_state,
            shuffle=True,
            stratify=train_val_df[stratify_col],
        )

        LOGGER.info(
            "Split complete. Train=%d, Val=%d, Test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

    @staticmethod
    def split_xy(frame: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Separates features and target for downstream training."""
        if target_column not in frame.columns:
            raise KeyError(f"Target column '{target_column}' not found")

        x = frame.drop(columns=[target_column])
        y = frame[target_column]
        return x, y
