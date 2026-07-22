"""T5 scoring — per-window anomaly scores for both detectors, one shared grid.

T5 compares two detectors that natively score at different granularities:

  * the LSTM-Autoencoder (T4) scores per WINDOW  — reconstruction MSE of a
    60-row window;
  * the z-score baseline (T3) scores per ROW     — how far one row's sensors
    deviate from the train-normal range.

A fair comparison needs both on the SAME axis. This module puts everything on a
single per-window grid:

  * window grid: sliding windows at ``stride=1`` (eval never skips a window, so
    no attack window is missed — the point of stride-1 at eval time);
  * AE score:   reconstruction MSE of each window (native);
  * baseline score: the row scores aggregated to the window by ``max`` over the
    rows inside it. ``max`` (not mean) is deliberate — it matches the window
    label semantics used everywhere in T5: a window is anomalous if ANY row in
    it is anomalous, so a window's baseline score is the strongest row deviation
    inside it. This keeps the baseline directly comparable to its own per-row
    flagging while living on the window grid;
  * labels: per-window OR of the row labels, computed for ``attack`` and each
    per-process column, so T5 can report per-process recall.

Every array returned here is aligned to the same window starts, so downstream
metrics (T5 File 2) index them interchangeably. This module computes scores
only — no thresholds, no metrics, no disk writes beyond the artifact loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader

from cps_anomaly_pipeline.baseline import SCORING_METHODS, BaselineModel
from cps_anomaly_pipeline.device import get_device
from cps_anomaly_pipeline.model_lstm_ae import LSTMAutoencoder, reconstruction_error
from cps_anomaly_pipeline.windowing import (
    DEFAULT_WINDOW,
    IntScaler,
    WindowDataset,
)

# Eval always uses stride 1: at scoring time we cannot afford to skip a window,
# or an attack that lives entirely inside a skipped window would go unseen.
# (Training subsamples with a larger stride for speed; evaluation does not.)
EVAL_STRIDE = 1


def window_starts(n_rows: int, window: int, stride: int) -> list[int]:
    """Start indices of every window, matching WindowDataset's convention.

    A window covers rows ``[s, s + window)``. The last start is ``n - window``.
    """
    if n_rows < window:
        raise ValueError(f"{n_rows} rows < window {window}")
    return list(range(0, n_rows - window + 1, stride))


def window_label_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    window: int = DEFAULT_WINDOW,
    stride: int = EVAL_STRIDE,
) -> dict[str, np.ndarray]:
    """Per-window label for each named column: 1 if ANY row in the window is 1.

    Columns absent from ``df`` (e.g. per-process labels on a split that lacks
    them) are skipped rather than faked. Uses a sliding-window max, which equals
    a logical OR on 0/1 columns and is vectorised (no Python per-window loop).
    """
    n = len(df)
    n_windows = len(window_starts(n, window, stride))
    out: dict[str, np.ndarray] = {}
    for col in columns:
        if col not in df.columns:
            continue
        row = df[col].to_numpy(dtype=np.int64)
        # sliding_window_view -> (n - window + 1, window); max over the window
        # axis is the OR for 0/1 data. Then subsample by stride.
        windowed = sliding_window_view(row, window).max(axis=1)[::stride]
        assert len(windowed) == n_windows
        out[col] = windowed.astype(np.int64)
    return out


def score_baseline(
    model: BaselineModel,
    df: pd.DataFrame,
    method: str,
    window: int = DEFAULT_WINDOW,
    stride: int = EVAL_STRIDE,
) -> np.ndarray:
    """Per-window baseline score = max of the row scores inside each window.

    The row scores come from the fitted (train-only) z-score model; aggregating
    by ``max`` lifts them onto the window grid without re-fitting anything, so
    the baseline stays the same naive detector T3 defined — just read per window.
    """
    row_scores = model.score(df, method)  # (n_rows,)
    if len(row_scores) < window:
        raise ValueError(f"{len(row_scores)} rows < window {window}")
    return sliding_window_view(row_scores, window).max(axis=1)[::stride]


def score_ae(
    model: LSTMAutoencoder,
    int_scaler: IntScaler,
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    stride: int = EVAL_STRIDE,
    device: str | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Per-window reconstruction MSE of the LSTM-AE, on the same window grid.

    Windows are produced by the same WindowDataset used in training, so the
    feature assembly (int-then-float, train-fit int scaler, Gold-scaled floats)
    is identical to what the model was trained on. Runs under ``no_grad`` in eval
    mode; device is auto-selected unless overridden.
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()

    dataset = WindowDataset(df, int_scaler, window=window, stride=stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            err = reconstruction_error(model, batch)  # (batch,)
            scores.append(err.detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)


@dataclass
class WindowedScores:
    """All per-window arrays for one split, aligned on identical window starts.

    ``starts`` and every score/label array share the same length and index, so
    ``labels['attack'][i]`` is the label of the window scored by ``ae[i]`` and
    ``baseline['max_abs_z'][i]``.
    """

    starts: np.ndarray  # (n_windows,) row index of each window start
    ae: np.ndarray  # (n_windows,) reconstruction MSE
    baseline: dict[str, np.ndarray]  # method -> (n_windows,) window score
    labels: dict[str, np.ndarray]  # label column -> (n_windows,) 0/1

    def __post_init__(self) -> None:
        n = len(self.starts)
        for name, arr in self.baseline.items():
            assert len(arr) == n, f"baseline[{name}] len {len(arr)} != {n}"
        for name, arr in self.labels.items():
            assert len(arr) == n, f"labels[{name}] len {len(arr)} != {n}"
        assert len(self.ae) == n, f"ae len {len(self.ae)} != {n}"


def score_split(
    ae_model: LSTMAutoencoder,
    int_scaler: IntScaler,
    baseline_model: BaselineModel,
    df: pd.DataFrame,
    label_columns: tuple[str, ...],
    window: int = DEFAULT_WINDOW,
    stride: int = EVAL_STRIDE,
    device: str | None = None,
) -> WindowedScores:
    """Score one Gold split with both detectors and attach per-window labels.

    Returns a WindowedScores bundle; all arrays are aligned to the same starts.
    """
    starts = np.array(window_starts(len(df), window, stride), dtype=np.int64)
    ae = score_ae(ae_model, int_scaler, df, window, stride, device)
    baseline = {
        method: score_baseline(baseline_model, df, method, window, stride)
        for method in SCORING_METHODS
    }
    labels = window_label_columns(df, label_columns, window, stride)
    return WindowedScores(starts=starts, ae=ae, baseline=baseline, labels=labels)


# --- Artifact loaders (thin; the runner in T5 File 3 owns paths) ------------


def load_lstm_ae(ckpt_path: Path | str, device: str | None = None) -> LSTMAutoencoder:
    """Rebuild the LSTM-AE and load a ``best.pt`` state_dict (weights only).

    ``best.pt`` is a state_dict (see train.py), so the architecture is
    reconstructed here with its defaults — the same defaults training used — and
    the weights are loaded into it.
    """
    device = device or get_device()
    model = LSTMAutoencoder()
    state = torch.load(Path(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_int_scaler(npz_path: Path | str) -> IntScaler:
    """Load the train-fit integer-actuator scaler saved next to the checkpoint.

    Reconstructs the exact IntScaler (columns / min / range) train.py persisted,
    so eval scales the int actuators identically to training — no re-fit, no
    leakage from calib/holdout.
    """
    data = np.load(Path(npz_path), allow_pickle=False)
    return IntScaler(
        columns=tuple(str(c) for c in data["columns"]),
        min_=data["min_"],
        range_=data["range_"],
    )
