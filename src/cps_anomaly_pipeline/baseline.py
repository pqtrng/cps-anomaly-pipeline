"""T3 baseline anomaly detector — z-score, no deep learning.

This is the comparison floor the LSTM-Autoencoder (T4) must beat. The idea is
deliberately simple: describe the normal operating range of every continuous
sensor with a per-sensor mean/std learned from TRAIN only, then score every row
by how far its sensors deviate from that range in standard-deviation units.

Two scoring methods are computed and kept side by side (compared in T5):
  * score_max_abs_z : max over sensors of |z|  -> reacts to a single sensor
                      swinging hard (a sharp, localised anomaly).
  * score_mean_sq_z : mean over sensors of z^2 -> reacts to many sensors
                      drifting together (a diffuse, system-wide anomaly).

Design rules (unsupervised, honest):
  * z is computed on the 55 continuous FLOAT sensors only. Integer actuators
    are discrete state, not Gaussian-distributed magnitudes, so a z-score is
    meaningless for them — they are excluded from scoring here.
  * mean/std are fit on TRAIN (normal-only) and reused verbatim for every other
    split. Labels are never consulted while scoring — they exist only for the
    T5 evaluation, not here.
  * the threshold is a percentile of the TRAIN scores (default p99.5). It is
    learned from train alone and applied unchanged to calib/holdout, so a row
    in holdout can never influence its own flag decision (no leakage).

This module produces flag counts only. Full metrics (per-attack recall, lead
time, point-adjust critique) are T5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cps_anomaly_pipeline.paths import PathConfig
from cps_anomaly_pipeline.schema import FLOAT_SENSOR_COLUMNS

# Guard against divide-by-zero: a sensor that is perfectly constant in train has
# std 0. Such a sensor carries no in-range variation to normalise against, so we
# floor its std to 1.0 — any deviation in test is then measured in raw units
# rather than producing inf/nan. (The known constant-in-train columns are int
# actuators already excluded here; this is a defensive floor, not a fix-up.)
_STD_FLOOR = 1e-8


DEFAULT_PERCENTILE = 99.5


@dataclass
class BaselineModel:
    """A fitted z-score baseline: per-sensor mean/std learned from train.

    Holds no labels and no threshold — scoring is a pure function of the sensor
    statistics. Thresholds are derived separately (see ``fit_threshold``) so the
    train-only provenance of the threshold is explicit at the call site.
    """

    columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "BaselineModel":
        """Learn per-sensor mean/std from TRAIN (normal-only) float sensors."""
        cols = FLOAT_SENSOR_COLUMNS
        values = train_df[list(cols)].to_numpy(dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        # Floor near-zero std so a constant sensor can't create inf/nan z-scores.
        std = np.where(std < _STD_FLOOR, 1.0, std)
        return cls(columns=cols, mean=mean, std=std)

    def _z(self, df: pd.DataFrame) -> np.ndarray:
        """Per-sensor z-scores for every row, using the TRAIN mean/std."""
        values = df[list(self.columns)].to_numpy(dtype=float)
        return (values - self.mean) / self.std

    def score_max_abs_z(self, df: pd.DataFrame) -> np.ndarray:
        """Row score = max over sensors of |z|. Sharp, localised deviations."""
        return np.abs(self._z(df)).max(axis=1)

    def score_mean_sq_z(self, df: pd.DataFrame) -> np.ndarray:
        """Row score = mean over sensors of z^2. Diffuse, system-wide drift."""
        return np.square(self._z(df)).mean(axis=1)

    def score(self, df: pd.DataFrame, method: str) -> np.ndarray:
        """Dispatch to a named scoring method."""
        if method == "max_abs_z":
            return self.score_max_abs_z(df)
        if method == "mean_sq_z":
            return self.score_mean_sq_z(df)
        raise ValueError(f"unknown scoring method: {method!r}")


SCORING_METHODS: tuple[str, ...] = ("max_abs_z", "mean_sq_z")


def fit_threshold(
    train_scores: np.ndarray, percentile: float = DEFAULT_PERCENTILE
) -> float:
    """Threshold = a percentile of TRAIN scores. Learned from train alone.

    A row is flagged anomalous when its score exceeds this value. Because the
    threshold comes only from train, no test/holdout row influences its own flag.
    """
    return float(np.percentile(train_scores, percentile))


def flag(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean flags: True where score strictly exceeds the threshold."""
    return scores > threshold


def run_baseline(
    paths: PathConfig | None = None, percentile: float = DEFAULT_PERCENTILE
) -> dict:
    """Fit on Gold train, score every split with both methods, report flag counts.

    Returns a nested summary dict (method -> split -> counts). No metrics beyond
    raw flag rates — evaluation against labels is T5.
    """
    paths = paths or PathConfig()
    gold_dir = paths.hai_gold_dir

    train = pd.read_parquet(gold_dir / "train.parquet")
    splits = {
        "train": train,
        "test_calib": pd.read_parquet(gold_dir / "test_calib.parquet"),
        "test_holdout": pd.read_parquet(gold_dir / "test_holdout.parquet"),
    }

    model = BaselineModel.fit(train)

    summary: dict = {}
    for method in SCORING_METHODS:
        train_scores = model.score(train, method)
        threshold = fit_threshold(train_scores, percentile)
        print(f"\n[{method}]  threshold (train p{percentile}) = {threshold:.4f}")
        summary[method] = {"threshold": threshold, "splits": {}}
        for name, df in splits.items():
            scores = model.score(df, method)
            flags = flag(scores, threshold)
            n_flag = int(flags.sum())
            rate = n_flag / len(df)
            summary[method]["splits"][name] = {
                "rows": len(df),
                "flagged": n_flag,
                "rate": rate,
            }
            print(
                f"  {name:14s} {len(df):>8,} rows  flagged={n_flag:>7,}  rate={rate:6.2%}"
            )
    return summary


def main() -> None:
    paths = PathConfig()
    print(f"Baseline z-score detector — reading Gold from: {paths.hai_gold_dir}")
    print(
        f"Scoring on {len(FLOAT_SENSOR_COLUMNS)} float sensors "
        f"(int actuators excluded from z)."
    )
    run_baseline(paths)
    print("\nDone. Flag counts only — full metrics are T5.")


if __name__ == "__main__":
    main()
