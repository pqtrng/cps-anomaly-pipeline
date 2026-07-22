"""Tests for the T5 runner (eval_report).

The runner touches disk (Gold parquet, a checkpoint), so these are plumbing
tests on TINY synthetic Gold with a randomly-initialised model in a fake run
dir — CPU-only, no real HAI, no GPU. They check that:
  1. run-dir resolution picks the latest run, honours an override, and errors
     cleanly when nothing is there;
  2. run_eval produces the expected results schema for all three detectors and
     writes the canonical results file;
  3. --no-crosscheck omits the cross-check block;
  4. the console table names every detector.

Numbers are meaningless here (random model) — only the wiring is under test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from cps_anomaly_pipeline.eval_report import (
    DETECTOR_ORDER,
    format_table,
    resolve_run_dir,
    run_eval,
)
from cps_anomaly_pipeline.model_lstm_ae import LSTMAutoencoder
from cps_anomaly_pipeline.paths import PathConfig
from cps_anomaly_pipeline.schema import (
    FLOAT_SENSOR_COLUMNS,
    INT_SENSOR_COLUMNS,
    LABEL_COLUMNS,
)
from cps_anomaly_pipeline.windowing import IntScaler

WINDOW = 60


def _sensor_frame(n: int, with_labels: bool) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data: dict[str, object] = {}
    for c in FLOAT_SENSOR_COLUMNS:
        data[c] = rng.normal(0.0, 1.0, size=n)
    for c in INT_SENSOR_COLUMNS:
        data[c] = [0, 10] * (n // 2)
    if with_labels:
        for c in LABEL_COLUMNS:
            data[c] = [0] * n
    return pd.DataFrame(data)


def _write_gold(gold_dir):
    gold_dir.mkdir(parents=True, exist_ok=True)
    train = _sensor_frame(200, with_labels=False)

    calib = _sensor_frame(150, with_labels=True)
    calib.loc[120:129, "attack"] = 1  # a short attack, leaves normal windows

    holdout = _sensor_frame(150, with_labels=True)
    holdout.loc[100:109, ["attack", "attack_P2"]] = 1  # one attack interval

    train.to_parquet(gold_dir / "train.parquet", index=False)
    calib.to_parquet(gold_dir / "test_calib.parquet", index=False)
    holdout.to_parquet(gold_dir / "test_holdout.parquet", index=False)
    return train


def _write_run(run_dir, train):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = LSTMAutoencoder()
    torch.save(model.state_dict(), run_dir / "best.pt")
    scaler = IntScaler.fit(train)
    np.savez(
        run_dir / "int_scaler.npz",
        columns=np.array(scaler.columns),
        min_=scaler.min_,
        range_=scaler.range_,
    )
    (run_dir / "metrics.json").write_text(json.dumps({"best_val_loss": 0.083}))


def test_resolve_run_dir_latest_and_override(tmp_path):
    runs = tmp_path / "runs"
    model_root = runs / "lstm_ae"
    (model_root / "lstm_ae-20260101-000000").mkdir(parents=True)
    (model_root / "lstm_ae-20260722-105138").mkdir(parents=True)
    # Latest by lexicographic (== chronological) order.
    assert resolve_run_dir(runs).name == "lstm_ae-20260722-105138"
    # Explicit override.
    override = model_root / "lstm_ae-20260101-000000"
    assert resolve_run_dir(runs, run_dir=override) == override


def test_resolve_run_dir_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_run_dir(tmp_path / "runs")  # no runs at all
    with pytest.raises(FileNotFoundError):
        resolve_run_dir(tmp_path / "runs", run_dir=tmp_path / "nope")


def _paths(tmp_path) -> PathConfig:
    return PathConfig(data_root=tmp_path / "data", runs_root=tmp_path / "runs")


def test_run_eval_schema_and_file(tmp_path):
    paths = _paths(tmp_path)
    train = _write_gold(paths.hai_gold_dir)
    run_dir = paths.runs_dir / "lstm_ae" / "lstm_ae-20260722-105138"
    _write_run(run_dir, train)

    results = run_eval(
        paths=paths, percentile=99.5, crosscheck=True, results_dir=tmp_path / "results"
    )

    # Metadata.
    assert results["window"] == WINDOW
    assert results["stride"] == 1
    assert results["checkpoint_best_val_loss"] == 0.083
    assert isinstance(results["eval_device"], str) and results["eval_device"]

    # Every detector present with the full report schema.
    assert set(results["detectors"].keys()) == set(DETECTOR_ORDER)
    for label in DETECTOR_ORDER:
        d = results["detectors"][label]
        for key in (
            "threshold",
            "window_metrics_raw",
            "window_metrics_point_adjust",
            "point_adjust_inflation",
            "per_process_recall",
            "detection",
            "threshold_crosscheck",
        ):
            assert key in d, f"{label} missing {key}"
        # Point-adjust recall never below raw recall.
        infl = d["point_adjust_inflation"]
        assert infl["recall_point_adjust"] >= infl["recall_raw"]
        # P4 is never reported (HAI has no P4 label).
        assert "attack_P4" not in d["per_process_recall"]

    # Canonical file written.
    out = json.loads((tmp_path / "results" / "eval_metrics.json").read_text())
    assert out["detectors"].keys() == results["detectors"].keys()


def test_run_eval_no_crosscheck(tmp_path):
    paths = _paths(tmp_path)
    train = _write_gold(paths.hai_gold_dir)
    run_dir = paths.runs_dir / "lstm_ae" / "run-a"
    _write_run(run_dir, train)

    results = run_eval(paths=paths, crosscheck=False, results_dir=tmp_path / "results")
    for label in DETECTOR_ORDER:
        assert "threshold_crosscheck" not in results["detectors"][label]


def test_run_eval_pins_device(tmp_path):
    """--device cpu is recorded, so the number is reproducible off the CI path."""
    paths = _paths(tmp_path)
    train = _write_gold(paths.hai_gold_dir)
    _write_run(paths.runs_dir / "lstm_ae" / "run-a", train)
    results = run_eval(paths=paths, device="cpu", results_dir=tmp_path / "results")
    assert results["eval_device"] == "cpu"


def test_format_table_names_detectors(tmp_path):
    paths = _paths(tmp_path)
    train = _write_gold(paths.hai_gold_dir)
    run_dir = paths.runs_dir / "lstm_ae" / "run-a"
    _write_run(run_dir, train)
    results = run_eval(paths=paths, results_dir=tmp_path / "results")
    table = format_table(results)
    for label in DETECTOR_ORDER:
        assert label in table
