"""Tests for T5 scoring — per-window scores on a shared grid.

Verifies the properties T5's fair comparison depends on:
  1. every array (starts, ae, baseline, labels) shares one aligned length;
  2. a window label is the OR of its rows, per column, including per-process;
  3. the baseline window score is exactly the max of row scores inside it;
  4. the AE window score has the right shape and is finite / non-negative;
  5. saved artifacts (best.pt state_dict, int_scaler.npz) round-trip to the
     same scores — proof the loaders reconstruct model + scaler faithfully;
  6. per-process label columns absent from a split are skipped, not faked.

Synthetic frames keep tests fast and independent of real Gold, matching the
style of test_windowing.py / test_baseline.py. A randomly-initialised model is
enough here — these tests check the scoring plumbing, not model quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from cps_anomaly_pipeline.baseline import SCORING_METHODS, BaselineModel
from cps_anomaly_pipeline.model_lstm_ae import LSTMAutoencoder
from cps_anomaly_pipeline.schema import (
    FLOAT_SENSOR_COLUMNS,
    INT_SENSOR_COLUMNS,
    LABEL_COLUMNS,
)
from cps_anomaly_pipeline.scoring import (
    load_int_scaler,
    load_lstm_ae,
    score_ae,
    score_baseline,
    score_split,
    window_label_columns,
    window_starts,
)
from cps_anomaly_pipeline.windowing import IntScaler

WINDOW = 60


def _make_gold(n: int, offset: float = 0.0, with_labels: bool = True) -> pd.DataFrame:
    """Gold-shaped frame: 55 float sensors ~N(offset,1) + 5 int actuators."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for c in FLOAT_SENSOR_COLUMNS:
        data[c] = rng.normal(offset, 1.0, size=n)
    for c in INT_SENSOR_COLUMNS:
        data[c] = [0, 10] * (n // 2)
    if with_labels:
        for c in LABEL_COLUMNS:
            data[c] = [0] * n
    return pd.DataFrame(data)


def test_all_arrays_aligned():
    """score_split returns starts/ae/baseline/labels all the same length."""
    df = _make_gold(120)
    int_scaler = IntScaler.fit(df)
    baseline = BaselineModel.fit(df)
    model = LSTMAutoencoder()

    ws = score_split(model, int_scaler, baseline, df, LABEL_COLUMNS, window=WINDOW)

    expected = 120 - WINDOW + 1  # 61 windows at stride 1
    assert len(ws.starts) == expected
    assert len(ws.ae) == expected
    for method in SCORING_METHODS:
        assert len(ws.baseline[method]) == expected
    for col in LABEL_COLUMNS:
        assert len(ws.labels[col]) == expected


def test_window_labels_or_including_per_process():
    """A single attacked row flips every window that contains it, per column."""
    df = _make_gold(90)
    df.loc[65, "attack"] = 1
    df.loc[65, "attack_P2"] = 1  # per-process column must OR independently

    labels = window_label_columns(df, LABEL_COLUMNS, window=WINDOW, stride=1)
    starts = window_starts(len(df), WINDOW, 1)

    # Window start s covers rows [s, s+59]; contains row 65 when 6 <= s <= 65.
    for i, s in enumerate(starts):
        contains_65 = s <= 65 <= s + WINDOW - 1
        assert labels["attack"][i] == int(contains_65)
        assert labels["attack_P2"][i] == int(contains_65)
    # attack_P1 / attack_P3 were never set -> all zero.
    assert labels["attack_P1"].sum() == 0
    assert labels["attack_P3"].sum() == 0


def test_baseline_window_score_is_max_over_rows():
    """The per-window baseline score equals the sliding max of the row scores."""
    df = _make_gold(100)
    model = BaselineModel.fit(df)
    for method in SCORING_METHODS:
        row_scores = model.score(df, method)
        expected = sliding_window_view(row_scores, WINDOW).max(axis=1)
        got = score_baseline(model, df, method, window=WINDOW, stride=1)
        np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_baseline_window_reacts_to_a_spike():
    """A single hugely-deviated row lifts exactly the windows that contain it."""
    train = _make_gold(400)
    model = BaselineModel.fit(train)
    df = _make_gold(100)
    df.loc[70, FLOAT_SENSOR_COLUMNS] = 50.0  # extreme spike on one row
    scores = score_baseline(model, df, "max_abs_z", window=WINDOW, stride=1)
    starts = window_starts(len(df), WINDOW, 1)
    containing = [i for i, s in enumerate(starts) if s <= 70 <= s + WINDOW - 1]
    not_containing = [i for i in range(len(starts)) if i not in containing]
    assert scores[containing].min() > scores[not_containing].max()


def test_ae_score_shape_and_finite():
    """AE window score: one value per window, finite and non-negative (MSE)."""
    df = _make_gold(150)
    int_scaler = IntScaler.fit(df)
    model = LSTMAutoencoder()
    scores = score_ae(model, int_scaler, df, window=WINDOW, stride=1, device="cpu")
    assert len(scores) == 150 - WINDOW + 1
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()  # mean of squared errors


def test_artifact_roundtrip(tmp_path):
    """best.pt + int_scaler.npz reload to a model/scaler giving identical scores."""
    df = _make_gold(120)
    int_scaler = IntScaler.fit(df)
    model = LSTMAutoencoder()

    torch.manual_seed(0)
    before = score_ae(model, int_scaler, df, window=WINDOW, stride=1, device="cpu")

    # Persist exactly as train.py does.
    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)
    npz = tmp_path / "int_scaler.npz"
    np.savez(
        npz,
        columns=np.array(int_scaler.columns),
        min_=int_scaler.min_,
        range_=int_scaler.range_,
    )

    reloaded_model = load_lstm_ae(ckpt, device="cpu")
    reloaded_scaler = load_int_scaler(npz)
    assert reloaded_scaler.columns == int_scaler.columns
    after = score_ae(
        reloaded_model, reloaded_scaler, df, window=WINDOW, stride=1, device="cpu"
    )
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-8)


def test_absent_process_labels_skipped():
    """A split carrying only 'attack' yields labels for 'attack' alone."""
    df = _make_gold(80, with_labels=False)
    df["attack"] = [0] * 80  # only the top-level label present
    labels = window_label_columns(df, LABEL_COLUMNS, window=WINDOW, stride=1)
    assert set(labels.keys()) == {"attack"}
