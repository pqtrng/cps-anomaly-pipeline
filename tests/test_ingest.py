"""Smoke tests for T1 HAI -> Bronze ingestion.

A tiny synthetic HAI-shaped dataset is built in a temp dir so the tests never
touch the real (several-hundred-MB) files and CI stays fast and hermetic.
"""
from __future__ import annotations

import gzip
import pandas as pd
import pytest
from cps_anomaly_pipeline.ingest import LABEL_COLUMNS, ingest_hai
from cps_anomaly_pipeline.paths import PathConfig
from pathlib import Path


def _write_gz_csv(path: Path, df: pd.DataFrame) -> None:
    with gzip.open(path, "wt", newline="") as f:
        df.to_csv(f, index=False)


@pytest.fixture()
def hai_like(tmp_path: Path) -> PathConfig:
    """Create a minimal HAI-shaped raw tree: one train file, one test file."""
    paths = PathConfig(data_root=tmp_path)
    raw = paths.hai_raw_dir
    raw.mkdir(parents=True)

    train = pd.DataFrame(
        {"time": range(5), "P1_FT01": [1.0] * 5, "P2_VT01": [2.0] * 5}
    )
    _write_gz_csv(raw / "train1.csv.gz", train)

    test = pd.DataFrame(
        {
            "time": range(3),
            "P1_FT01": [1.0] * 3,
            "P2_VT01": [2.0] * 3,
            "attack": [0, 1, 0],
            "attack_P1": [0, 1, 0],
            "attack_P2": [0, 0, 0],
            "attack_P3": [0, 0, 0],
        }
    )
    _write_gz_csv(raw / "test1.csv.gz", test)
    return paths


def test_bronze_files_written(hai_like: PathConfig) -> None:
    written = ingest_hai(hai_like)
    assert len(written) == 2
    assert all(p.exists() and p.suffix == ".parquet" for p in written)


def test_row_counts_preserved(hai_like: PathConfig) -> None:
    ingest_hai(hai_like)
    bronze = hai_like.hai_bronze_dir
    assert pd.read_parquet(bronze / "train1.parquet").shape[0] == 5
    assert pd.read_parquet(bronze / "test1.parquet").shape[0] == 3


def test_labels_only_in_test(hai_like: PathConfig) -> None:
    ingest_hai(hai_like)
    bronze = hai_like.hai_bronze_dir
    train_cols = set(pd.read_parquet(bronze / "train1.parquet").columns)
    test_cols = set(pd.read_parquet(bronze / "test1.parquet").columns)

    assert {"source_file", "ingested_at"} <= train_cols
    assert {"source_file", "ingested_at"} <= test_cols
    assert not (set(LABEL_COLUMNS) & train_cols)
    assert set(LABEL_COLUMNS) <= test_cols
