"""Gold layer builder for the HAI pipeline.

Silver -> Gold:
  1. Concatenate Silver files into three splits (see schema.GOLD_SPLITS):
       train         -> normal-only, model training
       test_calib    -> attack data allowed for threshold calibration
       test_holdout  -> attack data held out for final report / demo
  2. Standard-scale the 55 continuous float sensors. The scaler is fit on TRAIN
     ONLY, then applied to every split — no test data leaks into the scaler.
     Integer actuators and labels are left unscaled.
  3. Persist the scaler (joblib) and a gold_manifest.json recording the split
     assignment, the rationale, per-split row counts, attack-type coverage, and
     SHA-1 fingerprints.

The split is deliberately chosen (not random / not a plain time cut) so that
both calib and holdout retain all three attack types, especially the rare P3.
This choice is recorded in the manifest for reproducibility and honest reporting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cps_anomaly_pipeline.fingerprint import fingerprint
from cps_anomaly_pipeline.paths import PathConfig
from cps_anomaly_pipeline.schema import (
    FLOAT_SENSOR_COLUMNS,
    GOLD_SPLITS,
    LABEL_COLUMNS,
    SENSOR_COLUMNS,
)

SPLIT_RATIONALE = (
    "Files assigned so both test_calib and test_holdout retain all three attack "
    "types (P1/P2/P3). P3 is rare and appears only in test2/test4/test5; the "
    "split keeps P3 in both calib (test2,test4) and holdout (test5). Not random, "
    "not a plain time cut — chosen for attack-type coverage and reported honestly."
)


def _load_split(silver_dir: Path, stems: tuple[str, ...]) -> pd.DataFrame:
    frames = [pd.read_parquet(silver_dir / f"{s}.parquet") for s in stems]
    return pd.concat(frames, ignore_index=True)


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the continuous float sensors of TRAIN only."""
    scaler = StandardScaler()
    scaler.fit(train_df[list(FLOAT_SENSOR_COLUMNS)])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy with float sensors standardised; other columns untouched."""
    out = df.copy()
    out[list(FLOAT_SENSOR_COLUMNS)] = scaler.transform(df[list(FLOAT_SENSOR_COLUMNS)])
    return out


def _attack_coverage(df: pd.DataFrame) -> dict[str, int]:
    return {c: int(df[c].sum()) for c in LABEL_COLUMNS if c in df.columns}


def build_gold(paths: PathConfig | None = None) -> dict[str, Path]:
    """Build the three Gold splits with a train-fit scaler and a manifest.

    Returns a mapping of split name -> written parquet path.
    """
    paths = paths or PathConfig()
    silver_dir = paths.hai_silver_dir
    gold_dir = paths.hai_gold_dir
    gold_dir.mkdir(parents=True, exist_ok=True)

    if not any(silver_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"No Silver parquet in {silver_dir}. Build Silver first."
        )

    # Load splits.
    splits = {
        name: _load_split(silver_dir, stems) for name, stems in GOLD_SPLITS.items()
    }

    # Fit scaler on TRAIN ONLY, then apply to every split.
    scaler = fit_scaler(splits["train"])
    scaler_path = gold_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    written: dict[str, Path] = {}
    manifest_splits: dict[str, dict] = {}
    for name, df in splits.items():
        scaled = apply_scaler(df, scaler)
        out_path = gold_dir / f"{name}.parquet"
        scaled.to_parquet(out_path, engine="pyarrow", index=False)
        written[name] = out_path

        fp = fingerprint(scaled, list(SENSOR_COLUMNS))
        manifest_splits[name] = {
            "files": list(GOLD_SPLITS[name]),
            "rows": int(len(scaled)),
            "sha1": fp,
            "attack_coverage": _attack_coverage(df),
        }
        cov = manifest_splits[name]["attack_coverage"]
        print(f"  {name:14s} {len(scaled):>7,} rows  sha1={fp[:12]}  attacks={cov}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scaler": {
            "type": "StandardScaler",
            "fit_on": "train",
            "columns": list(FLOAT_SENSOR_COLUMNS),
            "path": scaler_path.name,
        },
        "split_rationale": SPLIT_RATIONALE,
        "splits": manifest_splits,
    }
    manifest_path = gold_dir / "gold_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {manifest_path.name}")
    return written


def verify_gold(paths: PathConfig | None = None) -> bool:
    """Verify Gold: each test split must contain every attack type (>0).

    Returns True if all checks pass; raises AssertionError otherwise.
    """
    paths = paths or PathConfig()
    gold_dir = paths.hai_gold_dir
    manifest = json.loads((gold_dir / "gold_manifest.json").read_text())

    ok = True
    for name in ("test_calib", "test_holdout"):
        cov = manifest["splits"][name]["attack_coverage"]
        for label in LABEL_COLUMNS:
            count = cov.get(label, 0)
            status = "ok" if count > 0 else "MISSING"
            if count == 0:
                ok = False
            print(f"  {name:14s} {label:12s} = {count:>5}  [{status}]")
    assert ok, "A test split is missing an attack type — split assignment invalid."
    print("verify_gold: all test splits retain every attack type.")
    return ok


def main() -> None:
    paths = PathConfig()
    print(f"Building Gold from: {paths.hai_silver_dir}")
    print(f"Writing Gold to:    {paths.hai_gold_dir}")
    build_gold(paths)
    print("Verifying Gold...")
    verify_gold(paths)
    print("Done.")


if __name__ == "__main__":
    main()
