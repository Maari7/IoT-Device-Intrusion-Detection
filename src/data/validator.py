"""Validation layer for schema checks and data quality reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from pandera.errors import SchemaErrors
    from data.schema.data_schema import IoTDatasetSchema, SchemaSettings
    _PANDERA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    SchemaErrors = Exception
    IoTDatasetSchema = None
    SchemaSettings = None
    _PANDERA_AVAILABLE = False

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

        self.schema = None
        self.feature_columns: List[str] = []

        if _PANDERA_AVAILABLE and IoTDatasetSchema is not None and SchemaSettings is not None:
            feature_columns = IoTDatasetSchema.infer_feature_columns(paths_cfg["benign_file"])
            self.schema = IoTDatasetSchema(
                feature_columns=feature_columns,
                settings=SchemaSettings(
                    label_column=self.label_column,
                    family_column=self.family_column,
                    source_column=self.source_column,
                ),
            )
            self.feature_columns = list(feature_columns)
        else:
            LOGGER.warning("Pandera is not available; using lightweight validator fallback")
            benign_path = Path(paths_cfg["benign_file"])
            if benign_path.exists():
                self.feature_columns = list(pd.read_csv(benign_path, nrows=0).columns)

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validates a labeled DataFrame and raises on schema mismatch."""
        LOGGER.info("Validating merged dataframe with shape %s", frame.shape)

        if frame.columns.duplicated().any():
            duplicated = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns found: {duplicated}")

        required_columns = [self.label_column, self.family_column, self.source_column]
        missing_required = [c for c in required_columns if c not in frame.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        if self.schema is not None:
            try:
                validated = self.schema.validate_labeled(frame)
            except SchemaErrors as exc:
                LOGGER.error("Schema validation failed with %d failure cases", len(exc.failure_cases))
                raise
        else:
            validated = frame.copy()

        if validated[self.label_column].isna().any():
            raise ValueError("Label column contains null values after validation")

        return validated

    def quality_report(self, frame: pd.DataFrame) -> Dict:
        """Builds a compact quality report for traceability."""
        feature_cols: List[str] = self.feature_columns
        if not feature_cols:
            metadata = {self.label_column, self.family_column, self.source_column, "label_id"}
            feature_cols = [c for c in frame.columns if c not in metadata]

        available_cols = [col for col in feature_cols if col in frame.columns]
        if len(available_cols) == 0:
            raise ValueError("No valid feature columns found in dataset")

        numeric_frame = frame[available_cols]

        report = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "feature_columns": len(available_cols),
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
