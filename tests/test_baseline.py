"""Tests for T3 baseline z-score detector.

Verifies the three properties the scope pins down:
  1. scores are monotone in the magnitude of deviation from train normal;
  2. the threshold is a function of TRAIN scores only (no holdout leakage);
  3. integer actuators never enter the z-score.

Synthetic frames keep the tests fast and independent of real Gold files, in the
same style as test_gold.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cps_anomaly_pipeline.baseline import (
    SCORING_METHODS,
    BaselineModel,
    fit_threshold,
    flag,
)
from cps_anomaly_pipeline.schema import (
    FLOAT_SENSOR_COLUMNS,
    INT_SENSOR_COLUMNS,
)


def _make_gold(n: int, offset: float = 0.0) -> pd.DataFrame:
    """Gold-shaped frame: float sensors (~N(offset,1)) + int actuators.

    Float columns already carry an offset so we can build train vs deviated
    test sets. Int actuators are 0/1 discrete state.
    """
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for c in FLOAT_SENSOR_COLUMNS:
        data[c] = rng.normal(offset, 1.0, size=n)
    for c in INT_SENSOR_COLUMNS:
        data[c] = [0, 1] * (n // 2)
    return pd.DataFrame(data)


def test_score_monotone_in_deviation():
    """A row deviating further from train normal must score higher, for BOTH
    methods. Build train ~N(0,1), then three single-row frames pushed 1, 5, 20
    std away on every float sensor."""
    train = _make_gold(2000, offset=0.0)
    model = BaselineModel.fit(train)

    def _row(shift: float) -> pd.DataFrame:
        row = {c: [shift] for c in FLOAT_SENSOR_COLUMNS}
        for c in INT_SENSOR_COLUMNS:
            row[c] = [0]
        return pd.DataFrame(row)

    for method in SCORING_METHODS:
        s1 = model.score(_row(1.0), method)[0]
        s5 = model.score(_row(5.0), method)[0]
        s20 = model.score(_row(20.0), method)[0]
        assert s1 < s5 < s20, f"{method} not monotone: {s1} !< {s5} !< {s20}"


def test_threshold_depends_only_on_train():
    """The threshold must be computable from train scores alone; feeding a wildly
    different holdout must NOT change it (the function never sees holdout)."""
    train = _make_gold(2000, offset=0.0)
    holdout = _make_gold(2000, offset=50.0)  # extreme, unrelated
    model = BaselineModel.fit(train)

    for method in SCORING_METHODS:
        train_scores = model.score(train, method)
        thr_a = fit_threshold(train_scores)
        # Recompute after also scoring holdout — threshold input is unchanged.
        _ = model.score(holdout, method)
        thr_b = fit_threshold(train_scores)
        assert thr_a == thr_b


def test_holdout_flagged_more_than_train():
    """Sanity: a heavily deviated split trips the train-fit threshold far more
    often than train itself — the detector reacts to deviation, not to labels."""
    train = _make_gold(2000, offset=0.0)
    deviated = _make_gold(2000, offset=5.0)
    model = BaselineModel.fit(train)

    for method in SCORING_METHODS:
        train_scores = model.score(train, method)
        threshold = fit_threshold(train_scores)
        train_rate = flag(train_scores, threshold).mean()
        deviated_rate = flag(model.score(deviated, method), threshold).mean()
        assert deviated_rate > train_rate


def test_int_actuators_excluded_from_z():
    """Changing only integer-actuator values must not change any score — z is
    computed on float sensors only."""
    train = _make_gold(1000, offset=0.0)
    model = BaselineModel.fit(train)

    base = _make_gold(500, offset=1.0)
    mutated = base.copy()
    for c in INT_SENSOR_COLUMNS:
        mutated[c] = mutated[c].max() + 999  # arbitrary large actuator change

    for method in SCORING_METHODS:
        np.testing.assert_array_equal(
            model.score(base, method), model.score(mutated, method)
        )


def test_threshold_percentile_relationship():
    """A higher percentile yields a higher (stricter) threshold and thus no more
    flags on train."""
    train = _make_gold(2000, offset=0.0)
    model = BaselineModel.fit(train)
    scores = model.score(train, "max_abs_z")
    thr_low = fit_threshold(scores, percentile=95.0)
    thr_high = fit_threshold(scores, percentile=99.5)
    assert thr_high >= thr_low
    assert flag(scores, thr_high).sum() <= flag(scores, thr_low).sum()


def test_constant_train_sensor_no_nan():
    """A float sensor that is constant in train (std 0) must not produce nan/inf
    scores — the std floor protects the z-score."""
    train = _make_gold(500, offset=0.0)
    train[FLOAT_SENSOR_COLUMNS[0]] = 7.0  # force a constant sensor
    model = BaselineModel.fit(train)
    scores = model.score(train, "mean_sq_z")
    assert np.isfinite(scores).all()
