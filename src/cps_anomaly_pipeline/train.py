"""T4 training — model-agnostic reconstruction trainer for CPS autoencoders.

Trains any reconstruction ``nn.Module`` (the LSTM-AE now; a Dense-AE ablation
later reuses this file unchanged) on the normal-only train split, selecting the
checkpoint by lowest VALIDATION loss — not by the last epoch and not by any
label-based metric. val_loss is the honest selection signal here: the model
never sees anomalies in training, so a held-out slice of normal data is the only
label-free way to catch overfitting (this mirrors the P1 finding that val_loss
beats a metric-based criterion for checkpoint selection).

Experiment tracking is TensorBoard, deliberately, not MLflow. P1 uses an MLflow
Registry with a live tracking server; P2 prefers self-contained TensorBoard
event files that live and die with the run, with versioning handled by Git plus
a committed metrics.json. Different tool for a different architecture, on purpose.

Device is auto-selected via get_device() so the same code runs unchanged across
GPU-enabled and CPU-only environments.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from cps_anomaly_pipeline.device import get_device
from cps_anomaly_pipeline.paths import PathConfig
from cps_anomaly_pipeline.windowing import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW,
    IntScaler,
    WindowDataset,
)


@dataclass
class TrainConfig:
    """All knobs for a training run. Logged verbatim so a run is reproducible."""

    model_name: str = "lstm_ae"
    window: int = DEFAULT_WINDOW
    stride: int = DEFAULT_STRIDE
    train_stride: int = 10  # subsample train windows to cut redundancy/epoch time
    val_stride: int = 10  # subsample val too; val_loss needs enough windows, not
    #                       every stride-1 window (stride-1 makes val dominate the
    #                       epoch and waste GPU time for no accuracy gain).
    batch_size: int = 256
    epochs: int = (
        100  # ceiling; early stopping (patience) usually cuts well before this
    )
    patience: int = 8  # stop if val_loss doesn't improve for this many epochs
    lr: float = 1e-3
    val_fraction: float = 0.1  # tail of train split held out as normal-only val
    seed: int = 42


def _split_train_val(
    train_df: pd.DataFrame, val_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: earliest rows train, final tail is validation.

    A time-ordered split (not random) is used because windows are temporal — a
    random split would leak near-identical neighbouring windows across the
    boundary. Both halves are normal-only (this is the train split).
    """
    n_val = int(len(train_df) * val_fraction)
    if n_val < 1:
        raise ValueError("val_fraction too small for this split size")
    return train_df.iloc[:-n_val], train_df.iloc[-n_val:]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """One pass over loader. If optimizer is given -> train, else -> eval.

    Returns the mean per-batch reconstruction loss.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n_batches = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            n_batches += 1
    return total / max(n_batches, 1)


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    config: TrainConfig,
    run_dir: Path,
    device: str | None = None,
) -> dict:
    """Train, checkpoint the lowest-val-loss weights, log to TensorBoard.

    Writes into run_dir:
      * best.pt      — state_dict of the lowest-val-loss epoch
      * metrics.json — config + best epoch + loss history (Git-versioned)
      * tensorboard event files (loss/train, loss/val per epoch)
    Returns the metrics dict.
    """
    device = device or get_device()
    torch.manual_seed(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    best_val = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    stopped_early = False
    history: list[dict] = []
    ckpt_path = run_dir / "best.pt"

    print(
        f"Training {config.model_name} on {device} | "
        f"train_windows={len(train_ds):,} val_windows={len(val_ds):,}"
    )
    start = time.time()
    last_epoch = -1
    for epoch in range(config.epochs):
        last_epoch = epoch
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device, None)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = "  <- best (checkpointed)"
        else:
            epochs_since_improve += 1
        print(f"  epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}{marker}")

        # Early stopping: stop once val_loss has not improved for `patience`
        # epochs. The best checkpoint is already saved, so nothing is lost — this
        # only avoids burning epochs after the model has stopped learning.
        if epochs_since_improve >= config.patience:
            stopped_early = True
            print(
                f"  early stop: no val improvement for {config.patience} epochs "
                f"(best @ epoch {best_epoch})"
            )
            break

    writer.close()
    elapsed = time.time() - start

    metrics = {
        "config": asdict(config),
        "device": device,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "stopped_early": stopped_early,
        "last_epoch": last_epoch,
        "elapsed_sec": round(elapsed, 1),
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"Done in {elapsed:.1f}s. Best val_loss={best_val:.6f} @ epoch {best_epoch}. "
        f"Checkpoint: {ckpt_path}"
    )
    return metrics


def build_datasets(
    paths: PathConfig, config: TrainConfig
) -> tuple[WindowDataset, WindowDataset, IntScaler]:
    """Load Gold train, fit int scaler on TRAIN, build train/val window datasets.

    The int scaler is fit on the training portion only and returned so the same
    scaler is reused for scoring every other split in T5 (no leakage).
    """
    gold_dir = paths.hai_gold_dir
    train_full = pd.read_parquet(gold_dir / "train.parquet")
    train_df, val_df = _split_train_val(train_full, config.val_fraction)

    int_scaler = IntScaler.fit(train_df)
    train_ds = WindowDataset(train_df, int_scaler, config.window, config.train_stride)
    val_ds = WindowDataset(val_df, int_scaler, config.window, config.val_stride)
    return train_ds, val_ds, int_scaler


def _build_model(model_name: str):
    """Construct a model by name. Extended when the Dense-AE ablation lands."""
    if model_name == "lstm_ae":
        from cps_anomaly_pipeline.model_lstm_ae import LSTMAutoencoder

        return LSTMAutoencoder()
    raise ValueError(f"unknown model: {model_name!r}")


def main() -> None:
    paths = PathConfig()
    config = TrainConfig()
    device = get_device()
    print(f"Device: {device}")
    print(f"Reading Gold from: {paths.hai_gold_dir}")

    train_ds, val_ds, int_scaler = build_datasets(paths, config)
    model = _build_model(config.model_name)

    run_dir = paths.runs_dir / config.model_name
    metrics = train_model(model, train_ds, val_ds, config, run_dir, device)

    # Persist the int scaler next to the checkpoint for reuse in T5 scoring.
    np.savez(
        run_dir / "int_scaler.npz",
        columns=np.array(int_scaler.columns),
        min_=int_scaler.min_,
        range_=int_scaler.range_,
    )
    print(f"Saved int scaler + metrics under {run_dir}")
    print(f"View curves:  tensorboard --logdir {paths.runs_dir}")
    _ = metrics


if __name__ == "__main__":
    main()
