"""Centralized path configuration for the CPS anomaly pipeline.

All data locations resolve through PathConfig so no machine-specific path is
hardcoded. Override the data root via the DATA_ROOT environment variable;
defaults to ./data relative to the current working directory.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "data")).expanduser().resolve()


def _default_runs_root() -> Path:
    return Path(os.environ.get("RUNS_ROOT", "runs")).expanduser().resolve()


@dataclass
class PathConfig:
    """Resolves every path the pipeline needs from a single data root.

    Layer directories follow a Medallion pattern:
    raw (as-downloaded) -> bronze (raw-as-ingested Parquet) -> silver -> gold.
    """

    data_root: Path = field(default_factory=_default_data_root)
    runs_root: Path = field(default_factory=_default_runs_root)

    # raw input
    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def hai_raw_dir(self) -> Path:
        """Directory holding the HAI 21.03 *.csv.gz files."""
        return self.raw_dir / "hai" / "hai-21.03"

    # medallion layers
    @property
    def bronze_dir(self) -> Path:
        return self.data_root / "bronze"

    @property
    def hai_bronze_dir(self) -> Path:
        return self.bronze_dir / "hai"

    @property
    def silver_dir(self) -> Path:
        return self.data_root / "silver"

    @property
    def hai_silver_dir(self) -> Path:
        return self.silver_dir / "hai"

    @property
    def gold_dir(self) -> Path:
        return self.data_root / "gold"

    @property
    def hai_gold_dir(self) -> Path:
        return self.gold_dir / "hai"

    @property
    def runs_dir(self) -> Path:
        return self.runs_root

    def ensure_dirs(self) -> None:
        self.hai_bronze_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_PATHS = PathConfig()
