"""Pandera schema definitions for the Danmini Doorbell IoT intrusion dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pandera as pa
from pandera import Check


@dataclass(frozen=True)
class SchemaSettings:
    """Container for schema column names used throughout the data layer."""

    label_column: str = "label"
    family_column: str = "attack_family"
    source_column: str = "source_file"


class IoTDatasetSchema:
    """Builds and validates Pandera schemas for raw and labeled IoT data.

    The schema is inferred from the dataset header to ensure compatibility with
    the specific feature set present in the provided Doorbell dataset.
    """

    def __init__(
        self,
        feature_columns: Iterable[str],
        settings: SchemaSettings | None = None,
    ) -> None:
        feature_cols = list(feature_columns)
        if not feature_cols:
            raise ValueError("feature_columns cannot be empty")

        self.feature_columns: List[str] = feature_cols
        self.settings = settings or SchemaSettings()

    @staticmethod
    def infer_feature_columns(csv_path: str | Path) -> List[str]:
        """Infers feature column names from the first line of a CSV file.

        Args:
            csv_path: Absolute or relative path to a representative CSV file.

        Returns:
            List of feature column names.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find CSV file: {path}")

        header_df = pd.read_csv(path, nrows=0)
        columns = list(header_df.columns)
        if not columns:
            raise ValueError(f"No columns found in CSV header for {path}")
        return columns

    def raw_schema(self) -> pa.DataFrameSchema:
        """Creates schema for source CSV rows before labels are appended."""
        columns = {
            col: pa.Column(
                dtype=float,
                coerce=True,
                nullable=True,
                checks=[Check.not_equal_to(float("inf")), Check.not_equal_to(float("-inf"))],
            )
            for col in self.feature_columns
        }
        return pa.DataFrameSchema(columns=columns, strict=True)

    def labeled_schema(self) -> pa.DataFrameSchema:
        """Creates schema for merged rows with labels and metadata columns."""
        columns = {
            col: pa.Column(
                dtype=float,
                coerce=True,
                nullable=True,
                checks=[Check.not_equal_to(float("inf")), Check.not_equal_to(float("-inf"))],
            )
            for col in self.feature_columns
        }
        columns[self.settings.label_column] = pa.Column(str, nullable=False)
        columns[self.settings.family_column] = pa.Column(str, nullable=False)
        columns[self.settings.source_column] = pa.Column(str, nullable=False)
        return pa.DataFrameSchema(columns=columns, strict=True)

    def validate_raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validates a raw feature-only DataFrame against the raw schema."""
        return self.raw_schema().validate(frame, lazy=True)

    def validate_labeled(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validates a labeled DataFrame against the full schema."""
        return self.labeled_schema().validate(frame, lazy=True)
