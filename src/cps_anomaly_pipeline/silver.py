"""Silver layer builder for the HAI pipeline.

Bronze -> Silver:
  1. Drop the 19 always-constant columns (see schema.DROP_CONSTANT_COLUMNS).
  2. Keep the 3 attack-signal columns that are constant in train but vary in test.
  3. Validate dtypes and label domain with Pandera (lazy: report all violations).
  4. Compute a SHA-1 row-level fingerprint for drift detection (no auto-heal).

Nothing is imputed or altered beyond the column drop — HAI has no NaNs (see
EDA.md). Validation failures raise; they are never silently repaired.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from cps_anomaly_pipeline.fingerprint import fingerprint
from cps_anomaly_pipeline.paths import PathConfig
from cps_anomaly_pipeline.schema import (
    DROP_CONSTANT_COLUMNS,
    LABEL_COLUMNS,
    SENSOR_COLUMNS,
    SILVER_TEST_SCHEMA,
    SILVER_TRAIN_SCHEMA,
)


def _is_test(name: str) -> bool:
    return name.startswith("test")


def build_silver_frame(bronze_df: pd.DataFrame, is_test: bool) -> pd.DataFrame:
    """Transform one Bronze frame into a validated Silver frame.

    Drops constant columns, validates with the matching schema, and returns the
    validated frame. Raises on schema violation.

    Train files carry all-zero label columns in the raw HAI data; these are
    dropped from train Silver (no information, and misleading to keep). Test
    files keep their labels.
    """
    present_drops = [c for c in DROP_CONSTANT_COLUMNS if c in bronze_df.columns]
    if not is_test:
        present_drops += [c for c in LABEL_COLUMNS if c in bronze_df.columns]
    silver = bronze_df.drop(columns=present_drops)

    schema = SILVER_TEST_SCHEMA if is_test else SILVER_TRAIN_SCHEMA
    try:
        silver = schema.validate(silver, lazy=True)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        raise ValueError(f"Silver validation failed: {exc}") from exc
    return silver


def build_silver(paths: PathConfig | None = None) -> list[Path]:
    """Build the Silver layer from every HAI Bronze parquet.

    Returns the list of Silver parquet paths written. Prints a per-file report
    including row count and SHA-1 fingerprint.
    """
    paths = paths or PathConfig()
    bronze_dir = paths.hai_bronze_dir
    silver_dir = paths.hai_silver_dir
    silver_dir.mkdir(parents=True, exist_ok=True)

    bronze_files = sorted(bronze_dir.glob("*.parquet"))
    if not bronze_files:
        raise FileNotFoundError(
            f"No Bronze parquet in {bronze_dir}. Run ingestion first."
        )

    fingerprint_cols = list(SENSOR_COLUMNS)
    written: list[Path] = []
    for bronze_path in bronze_files:
        df = pd.read_parquet(bronze_path)
        is_test = _is_test(bronze_path.name)

        silver = build_silver_frame(df, is_test=is_test)
        fp = fingerprint(silver, fingerprint_cols)

        out_path = silver_dir / bronze_path.name
        silver.to_parquet(out_path, engine="pyarrow", index=False)
        written.append(out_path)

        kind = "test " if is_test else "train"
        print(
            f"  {bronze_path.name:16s} [{kind}] {len(silver):>7,} rows  "
            f"cols={silver.shape[1]:>3}  sha1={fp[:12]}"
        )
    return written


def main() -> None:
    paths = PathConfig()
    print(f"Building Silver from: {paths.hai_bronze_dir}")
    print(f"Writing Silver to:    {paths.hai_silver_dir}")
    print(
        f"Dropping {len(DROP_CONSTANT_COLUMNS)} constant columns; "
        f"keeping {len(SENSOR_COLUMNS)} sensors + {len(LABEL_COLUMNS)} labels."
    )
    written = build_silver(paths)
    print(f"Done. {len(written)} Silver file(s) written.")


if __name__ == "__main__":
    main()
