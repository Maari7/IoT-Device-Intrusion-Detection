"""Validation layer for schema checks and data quality reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandera.errors import SchemaErrors

from data.schema.data_schema import IoTDatasetSchema, SchemaSettings

LOGGER = logging.getLogger(__name__)


class DataValidator:
    """Runs schema and quality validation over merged IoT data."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        data_cfg = config["data"]
        paths_cfg = config["paths"]

        self.label_column = data_cfg["label_column"]
        self.family_column = data_cfg["family_column"]
        self.source_column = data_cfg["source_column"]

        feature_columns = IoTDatasetSchema.infer_feature_columns(paths_cfg["benign_file"])
        self.schema = IoTDatasetSchema(
            feature_columns=feature_columns,
            settings=SchemaSettings(
                label_column=self.label_column,
                family_column=self.family_column,
                source_column=self.source_column,
            ),
        )

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validates a labeled DataFrame and raises on schema mismatch."""
        LOGGER.info("Validating merged dataframe with shape %s", frame.shape)

        if frame.columns.duplicated().any():
            duplicated = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns found: {duplicated}")

        try:
            validated = self.schema.validate_labeled(frame)
        except SchemaErrors as exc:
            LOGGER.error("Schema validation failed with %d failure cases", len(exc.failure_cases))
            raise

        if validated[self.label_column].isna().any():
            raise ValueError("Label column contains null values after validation")

        return validated

    def quality_report(self, frame: pd.DataFrame) -> Dict:
        """Builds a compact quality report for traceability."""
        feature_cols: List[str] = self.schema.feature_columns
        numeric_frame = frame[feature_cols]

        report = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "feature_columns": len(feature_cols),
            "missing_ratio": float(numeric_frame.isna().mean().mean()),
            "label_distribution": frame[self.label_column].value_counts(dropna=False).to_dict(),
            "family_distribution": frame[self.family_column].value_counts(dropna=False).to_dict(),
        }
        return report

    def write_quality_report(self, frame: pd.DataFrame, report_path: str | Path) -> Path:
        """Writes quality report to JSON."""
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.quality_report(frame)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Wrote data quality report to %s", path)
        return path
