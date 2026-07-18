"""Ingest HAI 21.03 raw CSV.gz files into the Bronze layer as Parquet.

Bronze is raw-as-ingested: no cleaning, no validation, no schema coercion.
The only additions are provenance columns (source_file, ingested_at). Train
files carry no attack labels; test files carry attack/attack_P1/attack_P2/
attack_P3 and those columns are preserved verbatim for later stages.

Usage:
    uv run python -m cps_anomaly_pipeline.ingest
    # or, via the script entry point:
    uv run cps-anomaly-pipeline
"""
from __future__ import annotations

import pandas as pd
from cps_anomaly_pipeline.paths import PathConfig
from datetime import datetime, timezone
from pathlib import Path

LABEL_COLUMNS = ("attack", "attack_P1", "attack_P2", "attack_P3")


def _ingested_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_hai_csv(path: Path) -> pd.Dataframe:
    """Read one HAI CSV.gz exactly as-is; only strip stray header whitespace."""
    df = pd.read_csv(path, compression="gzip")
    df.columns = [c.strip() for c in df.columns]
    return df


def ingest_file(csv_path: Path, bronze_dir: Path) -> Path:
    """Ingest a single HAI CSV.gz into one Bronze Parquet file.
    Provenance columns are appended; nothing else is altered. Returns the
    Bronze Parquet path written.
    """
    df = _read_hai_csv(csv_path)
    df["source_file"] = csv_path.name
    df["ingested_at"] = _ingested_at()

    out_name = csv_path.name.replace(".csv.gz", ".parquet")
    out_path = bronze_dir / out_name
    df.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


def ingest_hai(paths: PathConfig | None = None) -> list[Path]:
    """Ingest every HAI train*.csv.gz and test*.csv.gz into Bronze.

    Returns the list of Bronze Parquet paths written, in sorted input order.
    """
    paths = paths or PathConfig()
    paths.ensure_dirs()

    src_dir = paths.hai_raw_dir
    if not src_dir.exists():
        raise FileNotFoundError(f"HAI raw directory not found: {src_dir}. "
                                f"Set DATA_ROOT or place files under {src_dir}."
                                )

    csv_files = sorted(src_dir.glob("*.csv.gz"))
    if not csv_files:
        raise FileNotFoundError(f"No *.csv.gz files found in {src_dir}")

    written: list[Path] = []
    for csv_path in csv_files:
        out_path = ingest_file(csv_path, paths.hai_bronze_dir)
        n_rows = pd.read_parquet(out_path).shape[0]
        print(f" {csv_path.name:20s} -> {out_path.name:20s} ({n_rows:,} rows)")
        written.append(out_path)
    return written


def main() -> None:
    paths = PathConfig()
    print(f"Ingesting HAI from: {paths.hai_raw_dir}")
    print(f"Writing Bronze to: {paths.hai_bronze_dir}")
    written = ingest_hai(paths)
    print(f"Done. {len(written):,} Bronze files written.")


if __name__ == "__main__":
    main()
