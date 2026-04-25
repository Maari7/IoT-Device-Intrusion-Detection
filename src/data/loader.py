"""Dataset loading utilities with chunked CSV ingestion and optional parquet caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetFileSpec:
    """Metadata describing a single source CSV file."""

    path: Path
    label: str
    family: str


class DataLoader:
    """Loads benign and attack CSV files and appends metadata labels."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        paths_cfg = config["paths"]
        data_cfg = config["data"]

        self.dataset_root = Path(paths_cfg["dataset_root"])
        self.benign_file = Path(paths_cfg["benign_file"])
        self.attack_dirs = [Path(p) for p in paths_cfg["attack_dirs"]]
        self.raw_cache = Path(paths_cfg["raw_cache"])

        self.chunksize: int = int(data_cfg["chunksize"])
        self.cache_enabled: bool = bool(data_cfg["cache_enabled"])
        self.max_rows_per_file: Optional[int] = data_cfg.get("max_rows_per_file")

        self.label_column = data_cfg["label_column"]
        self.family_column = data_cfg["family_column"]
        self.source_column = data_cfg["source_column"]

    def discover_dataset_files(self) -> List[DatasetFileSpec]:
        """Discovers all benign and attack data files for ingestion."""
        specs: List[DatasetFileSpec] = []

        if not self.benign_file.exists():
            raise FileNotFoundError(f"Benign file not found: {self.benign_file}")

        specs.append(
            DatasetFileSpec(
                path=self.benign_file,
                label="benign",
                family="benign",
            )
        )

        for attack_dir in self.attack_dirs:
            if not attack_dir.exists():
                LOGGER.warning("Attack directory not found, skipping: %s", attack_dir)
                continue

            family = attack_dir.name.replace("_attacks", "")
            for csv_path in sorted(attack_dir.glob("*.csv")):
                label = f"{family}_{csv_path.stem}"
                specs.append(
                    DatasetFileSpec(
                        path=csv_path,
                        label=label,
                        family=family,
                    )
                )

        if len(specs) == 1:
            LOGGER.warning("Only benign file discovered. No attack files detected.")

        LOGGER.info("Discovered %d CSV files for ingestion.", len(specs))
        return specs

    def iter_labeled_chunks(
        self,
        spec: DatasetFileSpec,
        max_rows: Optional[int] = None,
    ) -> Iterator[pd.DataFrame]:
        """Yields labeled chunks from a source CSV.

        Args:
            spec: Source file metadata.
            max_rows: Optional row cap per file for faster iteration.
        """
        emitted_rows = 0
        for chunk in pd.read_csv(spec.path, chunksize=self.chunksize):
            chunk = chunk.astype("float32")
            chunk[self.label_column] = spec.label
            chunk[self.family_column] = spec.family
            chunk[self.source_column] = spec.path.name

            if max_rows is not None:
                remaining = max_rows - emitted_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            emitted_rows += len(chunk)
            yield chunk

            if max_rows is not None and emitted_rows >= max_rows:
                break

    def stream_dataset(self, max_rows_per_file: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Streams labeled rows for the full dataset as chunked DataFrames."""
        row_cap = max_rows_per_file if max_rows_per_file is not None else self.max_rows_per_file
        specs = self.discover_dataset_files()
        for spec in specs:
            LOGGER.info("Loading %s", spec.path)
            yield from self.iter_labeled_chunks(spec=spec, max_rows=row_cap)

    def load_dataset(
        self,
        use_cache: bool = True,
        max_rows_per_file: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads dataset into a single DataFrame, using cache when available."""
        if use_cache and self.cache_enabled and self.raw_cache.exists():
            LOGGER.info("Reading cached parquet from %s", self.raw_cache)
            return pd.read_parquet(self.raw_cache)

        frames: List[pd.DataFrame] = []
        for chunk in self.stream_dataset(max_rows_per_file=max_rows_per_file):
            frames.append(chunk)

        if not frames:
            raise RuntimeError("No data loaded. Check dataset paths and CSV files.")

        df = pd.concat(frames, axis=0, ignore_index=True)
        LOGGER.info("Loaded dataframe shape: %s", df.shape)

        if self.cache_enabled:
            self.raw_cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.raw_cache, index=False)
            LOGGER.info("Wrote raw cache parquet to %s", self.raw_cache)

        return df
