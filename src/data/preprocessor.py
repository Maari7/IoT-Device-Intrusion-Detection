"""Preprocessing pipeline for IoT IDS data ingestion and split generation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.loader import DataLoader
from src.data.splitter import DatasetSplitter, SplitConfig
from src.data.validator import DataValidator

LOGGER = logging.getLogger(__name__)


class Preprocessor:
    """Fits and applies preprocessing transformations for feature columns."""

    def __init__(
        self,
        label_column: str,
        family_column: str,
        source_column: str,
        impute_strategy: str = "median",
        clip_quantiles: tuple[float, float] = (0.001, 0.999),
        scale_features: bool = True,
    ) -> None:
        self.label_column = label_column
        self.family_column = family_column
        self.source_column = source_column

        self.impute_strategy = impute_strategy
        self.clip_quantiles = clip_quantiles
        self.scale_features = scale_features

        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        self.scaler = StandardScaler() if self.scale_features else None
        self.label_encoder = LabelEncoder()

        self.feature_columns: List[str] = []
        self.lower_bounds: pd.Series | None = None
        self.upper_bounds: pd.Series | None = None
        self.is_fitted = False

    def _metadata_columns(self) -> List[str]:
        return [self.label_column, self.family_column, self.source_column]

    def _extract_feature_columns(self, frame: pd.DataFrame) -> List[str]:
        metadata = set(self._metadata_columns())
        feature_columns = [c for c in frame.columns if c not in metadata and c != "label_id"]
        if not feature_columns:
            raise ValueError("No feature columns found for preprocessing")
        return feature_columns

    def _encode_labels(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            frame["label_id"] = self.label_encoder.fit_transform(frame[self.label_column])
        else:
            frame["label_id"] = self.label_encoder.transform(frame[self.label_column])
        return frame

    def fit(self, frame: pd.DataFrame) -> "Preprocessor":
        """Fits imputer, clipping bounds, scaler, and label encoder on train data."""
        self.feature_columns = self._extract_feature_columns(frame)

        numeric_train = frame[self.feature_columns].astype("float32")
        imputed = pd.DataFrame(
            self.imputer.fit_transform(numeric_train),
            columns=self.feature_columns,
            index=frame.index,
        )

        q_low, q_high = self.clip_quantiles
        self.lower_bounds = imputed.quantile(q_low)
        self.upper_bounds = imputed.quantile(q_high)
        clipped = imputed.clip(self.lower_bounds, self.upper_bounds, axis=1)

        if self.scaler is not None:
            self.scaler.fit(clipped)

        self.label_encoder.fit(frame[self.label_column])
        self.is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted transformations to an input split."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        if self.lower_bounds is None or self.upper_bounds is None:
            raise RuntimeError("Clipping bounds not initialized")

        numeric = frame[self.feature_columns].astype("float32")
        imputed = pd.DataFrame(
            self.imputer.transform(numeric),
            columns=self.feature_columns,
            index=frame.index,
        )

        clipped = imputed.clip(self.lower_bounds, self.upper_bounds, axis=1)

        if self.scaler is not None:
            scaled_values = self.scaler.transform(clipped)
            processed_features = pd.DataFrame(
                scaled_values,
                columns=self.feature_columns,
                index=frame.index,
            )
        else:
            processed_features = clipped

        metadata = frame[self._metadata_columns()].copy()
        transformed = pd.concat([processed_features, metadata], axis=1)
        transformed = self._encode_labels(transformed, fit=False)
        return transformed

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fits preprocessing artifacts and transforms train data."""
        self.fit(frame)
        transformed = self.transform(frame)
        return transformed

    def save(self, output_path: str | Path) -> Path:
        """Serializes preprocessor artifacts to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_columns": self.feature_columns,
            "lower_bounds": self.lower_bounds,
            "upper_bounds": self.upper_bounds,
            "config": {
                "label_column": self.label_column,
                "family_column": self.family_column,
                "source_column": self.source_column,
                "impute_strategy": self.impute_strategy,
                "clip_quantiles": self.clip_quantiles,
                "scale_features": self.scale_features,
            },
        }
        joblib.dump(payload, path)
        LOGGER.info("Saved preprocessor artifact to %s", path)
        return path


def setup_logging(level: str) -> None:
    """Configures process-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(config_path: str | Path) -> Dict:
    """Loads YAML configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_phase1(config: Dict, max_rows_per_file: int | None = None) -> Dict[str, Path]:
    """Runs the complete Phase 1 data pipeline end to end."""
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    split_cfg = config["split"]
    out_cfg = config["output"]

    processed_dir = Path(paths_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(config)
    validator = DataValidator(config)

    LOGGER.info("Loading merged dataset...")
    data = loader.load_dataset(max_rows_per_file=max_rows_per_file)
    # Prefer filtering by configured source column when device name is present in source paths.
    target_device = Path(paths_cfg.get("dataset_root", "")).name
    source_column = str(data_cfg.get("source_column", "source_file"))
    if target_device and source_column in data.columns:
        filtered = data[data[source_column].astype(str).str.contains(target_device, na=False)]
        if not filtered.empty:
            data = filtered
        else:
            LOGGER.warning("Source-based device filter produced no rows; proceeding without device filter")
    else:
        LOGGER.warning("No source column found for device filtering; proceeding without device filter")

    # Force a bounded training set size to keep preprocessing/training runtime practical.
    sample_cap = int(data_cfg.get("max_total_rows", 15000))
    sample_size = min(sample_cap, len(data))
    data = data.sample(n=sample_size, random_state=42)

    LOGGER.info(f"After filtering: {data.shape}")

    LOGGER.info("Validating loaded dataset...")
    validated = validator.validate(data)

    # Optional feature cap to reduce downstream training/SHAP cost.
    max_feature_columns = int(data_cfg.get("max_feature_columns", 30))
    metadata_columns = [
        str(data_cfg["label_column"]),
        str(data_cfg["family_column"]),
        str(data_cfg["source_column"]),
    ]
    feature_columns = [c for c in validated.columns if c not in metadata_columns and c != "label_id"]
    if max_feature_columns > 0 and len(feature_columns) > max_feature_columns:
        kept_features = feature_columns[:max_feature_columns]
        validated = validated[kept_features + metadata_columns]

        # 🔥 SAVE FEATURE LIST (CRITICAL FIX)
        feature_path = Path("artifacts/selected_features.joblib")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(kept_features, feature_path)

        LOGGER.info(f"Saved selected features: {len(kept_features)}")
        LOGGER.info(
            "Applied feature cap: kept %d of %d feature columns",
            len(kept_features),
            len(feature_columns),
        )

    report_path = processed_dir / "data_quality_report.json"
    validator.write_quality_report(validated, report_path)

    splitter = DatasetSplitter(
        SplitConfig(
            test_size=float(split_cfg["test_size"]),
            val_size=float(split_cfg["val_size"]),
            random_state=int(split_cfg["random_state"]),
            stratify_column=str(split_cfg["stratify_column"]),
        )
    )
    splits = splitter.split(validated)

    preprocessor = Preprocessor(
        label_column=data_cfg["label_column"],
        family_column=data_cfg["family_column"],
        source_column=data_cfg["source_column"],
        impute_strategy=prep_cfg["impute_strategy"],
        clip_quantiles=tuple(prep_cfg["clip_quantiles"]),
        scale_features=bool(prep_cfg["scale_features"]),
    )

    train_processed = preprocessor.fit_transform(splits["train"])
    val_processed = preprocessor.transform(splits["val"])
    test_processed = preprocessor.transform(splits["test"])

    train_path = processed_dir / out_cfg["train_file"]
    val_path = processed_dir / out_cfg["val_file"]
    test_path = processed_dir / out_cfg["test_file"]

    train_processed.to_parquet(train_path, index=False)
    val_processed.to_parquet(val_path, index=False)
    test_processed.to_parquet(test_path, index=False)

    LOGGER.info("Wrote processed train split to %s", train_path)
    LOGGER.info("Wrote processed val split to %s", val_path)
    LOGGER.info("Wrote processed test split to %s", test_path)

    artifact_path = Path(paths_cfg["features_dir"]) / out_cfg["preprocessor_file"]
    preprocessor.save(artifact_path)

    stats = {
        "train_rows": int(len(train_processed)),
        "val_rows": int(len(val_processed)),
        "test_rows": int(len(test_processed)),
        "feature_count": int(len(preprocessor.feature_columns)),
    }
    stats_path = processed_dir / "phase1_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    LOGGER.info("Wrote phase1 stats to %s", stats_path)

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "report": report_path,
        "stats": stats_path,
        "preprocessor": artifact_path,
    }


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the phase-1 pipeline."""
    parser = argparse.ArgumentParser(description="Run Phase 1 data pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional row cap per source CSV for smoke runs",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Phase 1 execution."""
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    outputs = run_phase1(config=config, max_rows_per_file=args.max_rows_per_file)
    LOGGER.info("Phase 1 complete. Outputs: %s", outputs)


if __name__ == "__main__":
    main()
