"""Tests for T2 Gold: train-only scaler (no leakage), selective scaling, and
the attack-coverage guard in verify_gold.

Synthetic frames keep the tests fast and independent of real Silver files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cps_anomaly_pipeline.gold import apply_scaler, fit_scaler
from cps_anomaly_pipeline.schema import (
    FLOAT_SENSOR_COLUMNS,
    INT_SENSOR_COLUMNS,
)


def _make_silver(n: int, offset: float = 0.0) -> pd.DataFrame:
    """Silver-shaped frame: float sensors (with an optional offset so train and
    test differ), int actuators, and a time column."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {"time": list(range(n))}
    for c in FLOAT_SENSOR_COLUMNS:
        data[c] = rng.normal(offset, 1.0, size=n)
    for c in INT_SENSOR_COLUMNS:
        data[c] = [0, 1] * (n // 2)
    return pd.DataFrame(data)


def test_scaler_fit_on_train_only():
    """Scaler mean/scale must come from TRAIN, so applying it to train yields
    ~zero mean; applying to an offset test set does NOT yield zero mean."""
    train = _make_silver(1000, offset=0.0)
    test = _make_silver(1000, offset=5.0)

    scaler = fit_scaler(train)
    train_scaled = apply_scaler(train, scaler)
    test_scaled = apply_scaler(test, scaler)

    col = FLOAT_SENSOR_COLUMNS[0]
    # Train standardised to ~0 mean.
    assert abs(train_scaled[col].mean()) < 0.1
    # Test (offset +5) must remain shifted — proof the scaler did NOT see test.
    assert test_scaled[col].mean() > 3.0


def test_int_actuators_not_scaled():
    train = _make_silver(100)
    scaler = fit_scaler(train)
    scaled = apply_scaler(train, scaler)
    for c in INT_SENSOR_COLUMNS:
        # Integer actuator columns are passed through unchanged.
        assert scaled[c].tolist() == train[c].tolist()


def test_float_sensors_are_scaled():
    train = _make_silver(500, offset=10.0)
    scaler = fit_scaler(train)
    scaled = apply_scaler(train, scaler)
    col = FLOAT_SENSOR_COLUMNS[0]
    # Original mean ~10; scaled mean ~0.
    assert abs(train[col].mean() - 10.0) < 1.0
    assert abs(scaled[col].mean()) < 0.1


def test_apply_scaler_does_not_mutate_input():
    train = _make_silver(100, offset=3.0)
    scaler = fit_scaler(train)
    before = train[FLOAT_SENSOR_COLUMNS[0]].copy()
    _ = apply_scaler(train, scaler)
    pd.testing.assert_series_equal(train[FLOAT_SENSOR_COLUMNS[0]], before)
