"""Tests for T4 windowing Dataset.

Checks the properties that matter for a correct, leakage-free windowed feed:
  1. window shape is (window, 60) and count matches the sliding-window formula;
  2. the int min-max scaler is fit on TRAIN only (offset test data is not
     squashed into train's [0,1] range);
  3. a window's label is the OR of its rows' attack flags;
  4. feature order is int-then-float, matching schema.SENSOR_COLUMNS;
  5. a train split (no label columns) yields all-zero window labels.

Synthetic frames keep tests fast and independent of real Gold files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cps_anomaly_pipeline.schema import (
    FLOAT_SENSOR_COLUMNS,
    INT_SENSOR_COLUMNS,
    SENSOR_COLUMNS,
)
from cps_anomaly_pipeline.windowing import (
    N_FEATURES,
    IntScaler,
    WindowDataset,
    build_feature_matrix,
)


def _make_gold(n: int, offset: float = 0.0, with_labels: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for c in FLOAT_SENSOR_COLUMNS:
        data[c] = rng.normal(offset, 1.0, size=n)
    for c in INT_SENSOR_COLUMNS:
        data[c] = [0, 10] * (n // 2)
    if with_labels:
        for c in ("attack", "attack_P1", "attack_P2", "attack_P3"):
            data[c] = [0] * n
    return pd.DataFrame(data)


def test_window_shape_and_count():
    df = _make_gold(100, with_labels=True)
    scaler = IntScaler.fit(df)
    ds = WindowDataset(df, scaler, window=60, stride=1)
    # sliding windows: n - window + 1 = 100 - 60 + 1 = 41
    assert len(ds) == 41
    item = ds[0]
    assert tuple(item.shape) == (60, N_FEATURES)
    assert item.dtype.is_floating_point


def test_int_scaler_fit_on_train_only():
    """Int min-max fit on train; an offset test set must NOT be forced into
    train's [0,1] — proof the scaler never saw test."""
    train = _make_gold(200)
    scaler = IntScaler.fit(train)
    train_scaled = scaler.transform(train)
    # Train actuators span {0,10} -> scaled to {0,1}.
    assert train_scaled.min() >= -1e-9
    assert train_scaled.max() <= 1.0 + 1e-9

    test = train.copy()
    for c in INT_SENSOR_COLUMNS:
        test[c] = test[c] + 100  # far outside train range
    test_scaled = scaler.transform(test)
    # Must exceed 1.0 — the train-fit scaler does not clamp test.
    assert test_scaled.max() > 5.0


def test_window_label_is_or_of_rows():
    df = _make_gold(80, with_labels=True)
    # Mark a single attacked row at index 65.
    df.loc[65, "attack"] = 1
    scaler = IntScaler.fit(df)
    ds = WindowDataset(df, scaler, window=60, stride=1)
    # Windows starting at 6..20 (inclusive) contain row 65 -> label 1.
    # start s covers rows [s, s+59]; contains 65 when s <= 65 <= s+59 -> 6<=s<=65.
    labels = ds.labels
    assert labels[6] == 1
    assert labels[5] == 0  # window [5,64] does not include row 65
    assert labels.sum() >= 1


def test_feature_order_int_then_float():
    df = _make_gold(70, with_labels=True)
    scaler = IntScaler.fit(df)
    matrix = build_feature_matrix(df, scaler)
    assert matrix.shape == (70, N_FEATURES)
    # First 5 columns are the int actuators (scaled to [0,1]); check the first
    # column equals the min-max of the first INT column.
    first_int = df[INT_SENSOR_COLUMNS[0]].to_numpy(dtype=float)
    expected = (first_int - first_int.min()) / (first_int.max() - first_int.min())
    np.testing.assert_allclose(matrix[:, 0], expected, rtol=1e-6)
    # Column count = 5 int + 55 float, in SENSOR_COLUMNS order.
    assert len(SENSOR_COLUMNS) == N_FEATURES


def test_train_split_all_zero_labels():
    """Train has no label columns -> every window label is 0."""
    train = _make_gold(90, with_labels=False)
    scaler = IntScaler.fit(train)
    ds = WindowDataset(train, scaler, window=60, stride=1)
    assert ds.labels.sum() == 0
    assert len(ds.labels) == len(ds)


def test_constant_int_actuator_no_nan():
    """An actuator constant in train (range 0) must not produce nan after scaling."""
    train = _make_gold(80)
    train[INT_SENSOR_COLUMNS[0]] = 3  # constant
    scaler = IntScaler.fit(train)
    scaled = scaler.transform(train)
    assert np.isfinite(scaled).all()
