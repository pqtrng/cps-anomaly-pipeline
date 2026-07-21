"""T4 windowing — sliding-window Dataset for the LSTM-Autoencoder.

The autoencoder consumes fixed-length temporal windows, not single rows. This
module turns a Gold split (per-row Parquet) into overlapping windows on the fly
inside a ``torch.utils.data.Dataset`` — no windowed Parquet is materialised, so
the Gold layer stays exactly as T2/T3 left it.

Feature set (60 columns, order fixed by schema.SENSOR_COLUMNS = int then float):
  * 55 continuous float sensors — already StandardScaler'd in Gold (train ~N(0,1)).
  * 5 integer actuators — NOT scaled in Gold, and on much larger raw ranges
    (e.g. P4_ST_PS in {0,50}). Left raw, their magnitude would dominate the MSE
    reconstruction loss and drown the float signal (EDA.md warns of exactly this).
    So they are min-max scaled to [0, 1] here, with the min/max learned on TRAIN
    ONLY and reused for every split — same no-leakage rule as the Gold scaler.
    Min-max (not z-score) because these are discrete state, not Gaussian
    magnitudes; z-score is meaningless for them (the same reason T3 excluded them).

A window is anomalous, for later evaluation, if ANY row inside it is attacked.
Labels are attached to windows here but are NOT used for training — the AE trains
on train (normal-only, no labels). Labels exist only so T5 can evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cps_anomaly_pipeline.schema import (
    FLOAT_SENSOR_COLUMNS,
    INT_SENSOR_COLUMNS,
    SENSOR_COLUMNS,
)

DEFAULT_WINDOW = 60  # 60 rows = 60 s at HAI's 1 Hz sampling.
DEFAULT_STRIDE = 1

# Feature order the model sees: int actuators first, then float sensors,
# matching schema.SENSOR_COLUMNS so every stage agrees on column order.
FEATURE_COLUMNS: tuple[str, ...] = SENSOR_COLUMNS
N_FEATURES = len(FEATURE_COLUMNS)  # 60


@dataclass
class IntScaler:
    """Min-max scaler for the integer actuators, fit on TRAIN only.

    Stored explicitly (not folded into the Dataset) so its train-only provenance
    is visible at the call site and it can be persisted alongside the model.
    """

    columns: tuple[str, ...]
    min_: np.ndarray
    range_: np.ndarray  # max - min, floored to avoid divide-by-zero

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "IntScaler":
        cols = INT_SENSOR_COLUMNS
        values = train_df[list(cols)].to_numpy(dtype=float)
        vmin = values.min(axis=0)
        vmax = values.max(axis=0)
        rng = vmax - vmin
        # A constant-in-train actuator has range 0 -> floor to 1 so it maps to a
        # constant 0 after scaling instead of producing nan.
        rng = np.where(rng == 0.0, 1.0, rng)
        return cls(columns=cols, min_=vmin, range_=rng)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        values = df[list(self.columns)].to_numpy(dtype=float)
        return (values - self.min_) / self.range_


def build_feature_matrix(df: pd.DataFrame, int_scaler: IntScaler) -> np.ndarray:
    """Assemble the (n_rows, 60) feature matrix in FEATURE_COLUMNS order.

    Float columns are taken as-is (already Gold-scaled); int actuators are
    min-max scaled with the train-fit scaler. The two blocks are concatenated in
    int-then-float order to match FEATURE_COLUMNS.
    """
    ints = int_scaler.transform(df)  # (n, 5) in schema INT order
    floats = df[list(FLOAT_SENSOR_COLUMNS)].to_numpy(dtype=float)  # (n, 55)
    matrix = np.concatenate([ints, floats], axis=1).astype(np.float32)
    assert matrix.shape[1] == N_FEATURES
    return matrix


def _window_labels(df: pd.DataFrame, window: int, stride: int) -> np.ndarray:
    """Per-window label: 1 if ANY row in the window is attacked, else 0.

    Returns an all-zero array when the split has no label columns (train).
    """
    n = len(df)
    starts = range(0, n - window + 1, stride)
    if "attack" not in df.columns:
        return np.zeros(len(list(starts)), dtype=np.int64)
    attack = df["attack"].to_numpy(dtype=np.int64)
    return np.array([int(attack[s : s + window].any()) for s in starts], dtype=np.int64)


class WindowDataset(Dataset):
    """Sliding windows over a Gold split, produced on the fly.

    Each item is a (window, 60) float32 tensor. The target of the autoencoder is
    the input itself (reconstruction), so no separate y is returned for training;
    the per-window attack label is exposed via ``labels`` for T5 evaluation only.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        int_scaler: IntScaler,
        window: int = DEFAULT_WINDOW,
        stride: int = DEFAULT_STRIDE,
    ) -> None:
        if len(df) < window:
            raise ValueError(f"split has {len(df)} rows, fewer than window={window}")
        self.window = window
        self.stride = stride
        self._matrix = build_feature_matrix(df, int_scaler)
        self._starts = list(range(0, len(df) - window + 1, stride))
        self.labels = _window_labels(df, window, stride)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        s = self._starts[idx]
        chunk = self._matrix[s : s + self.window]
        return torch.from_numpy(chunk)
