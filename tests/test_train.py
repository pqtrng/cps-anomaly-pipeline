"""Tests for T4 training loop (model-agnostic).

Kept tiny (2 epochs, small windows, small model) so they run on CPU in seconds:
  1. chronological train/val split keeps order and sizes;
  2. a short run writes best.pt + metrics.json and reports a best epoch;
  3. the checkpointed epoch is the one with the lowest val_loss in history;
  4. the loop trains an arbitrary reconstruction module (model-agnostic), shown
     with a trivial linear autoencoder standing in for the real models.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from cps_anomaly_pipeline.schema import FLOAT_SENSOR_COLUMNS, INT_SENSOR_COLUMNS
from cps_anomaly_pipeline.train import (
    TrainConfig,
    _split_train_val,
    train_model,
)
from cps_anomaly_pipeline.windowing import N_FEATURES, IntScaler, WindowDataset
from torch import nn


class _TinyAE(nn.Module):
    """Minimal per-timestep linear autoencoder — enough to exercise the loop."""

    def __init__(self, n_features: int = N_FEATURES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_gold(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for c in FLOAT_SENSOR_COLUMNS:
        data[c] = rng.normal(0.0, 1.0, size=n)
    for c in INT_SENSOR_COLUMNS:
        data[c] = [0, 10] * (n // 2)
    return pd.DataFrame(data)


def test_chronological_split_preserves_order():
    df = _make_gold(100)
    train, val = _split_train_val(df, val_fraction=0.2)
    assert len(train) == 80
    assert len(val) == 20
    # val is the final tail, in original order
    assert val.index[0] == 80
    assert val.index[-1] == 99


def test_short_run_writes_artifacts(tmp_path):
    df = _make_gold(200)
    train_df, val_df = _split_train_val(df, 0.2)
    scaler = IntScaler.fit(train_df)
    train_ds = WindowDataset(train_df, scaler, window=20, stride=5)
    val_ds = WindowDataset(val_df, scaler, window=20, stride=5)

    config = TrainConfig(
        model_name="tiny", window=20, stride=5, epochs=2, batch_size=16
    )
    metrics = train_model(_TinyAE(), train_ds, val_ds, config, tmp_path, device="cpu")

    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert metrics["best_epoch"] >= 0
    assert len(metrics["history"]) == 2


def test_checkpoint_is_lowest_val_loss(tmp_path):
    df = _make_gold(200)
    train_df, val_df = _split_train_val(df, 0.2)
    scaler = IntScaler.fit(train_df)
    train_ds = WindowDataset(train_df, scaler, window=20, stride=5)
    val_ds = WindowDataset(val_df, scaler, window=20, stride=5)

    config = TrainConfig(
        model_name="tiny", window=20, stride=5, epochs=4, batch_size=16
    )
    metrics = train_model(_TinyAE(), train_ds, val_ds, config, tmp_path, device="cpu")

    val_losses = [h["val_loss"] for h in metrics["history"]]
    best_by_history = min(range(len(val_losses)), key=lambda i: val_losses[i])
    assert metrics["best_epoch"] == best_by_history
    assert metrics["best_val_loss"] == min(val_losses)


def test_metrics_json_roundtrips(tmp_path):
    df = _make_gold(160)
    train_df, val_df = _split_train_val(df, 0.25)
    scaler = IntScaler.fit(train_df)
    train_ds = WindowDataset(train_df, scaler, window=20, stride=5)
    val_ds = WindowDataset(val_df, scaler, window=20, stride=5)

    config = TrainConfig(
        model_name="tiny", window=20, stride=5, epochs=2, batch_size=16
    )
    train_model(_TinyAE(), train_ds, val_ds, config, tmp_path, device="cpu")

    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert loaded["config"]["model_name"] == "tiny"
    assert "history" in loaded and len(loaded["history"]) == 2


class _IdentityAE(nn.Module):
    """Returns the input unchanged -> reconstruction loss is ~0 from epoch 0 and
    cannot improve further, so early stopping fires after `patience` epochs.
    A grad-requiring dummy param keeps backward() valid (its grad is 0)."""

    def __init__(self, n_features: int = N_FEATURES) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # identity plus 0*dummy so the loss has a (zero-gradient) path to a param
        return x + self._dummy * 0.0


def test_early_stopping_triggers(tmp_path):
    """With an identity model, val_loss is minimal at epoch 0 and never improves,
    so training stops after `patience` epochs instead of running all `epochs`."""
    df = _make_gold(200)
    train_df, val_df = _split_train_val(df, 0.2)
    scaler = IntScaler.fit(train_df)
    train_ds = WindowDataset(train_df, scaler, window=20, stride=5)
    val_ds = WindowDataset(val_df, scaler, window=20, stride=5)

    config = TrainConfig(
        model_name="identity", window=20, stride=5, epochs=50, patience=3, batch_size=16
    )
    metrics = train_model(_IdentityAE(), train_ds, val_ds, config, tmp_path, device="cpu")

    assert metrics["stopped_early"] is True
    assert metrics["last_epoch"] < 49
    assert len(metrics["history"]) == metrics["last_epoch"] + 1
