"""Tests for T2 Silver: column handling, schema validation, fingerprinting.

Uses small synthetic HAI-shaped frames so tests are fast and hermetic — no
dependency on the real Bronze parquet files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cps_anomaly_pipeline.fingerprint import fingerprint
from cps_anomaly_pipeline.silver import build_silver_frame
from cps_anomaly_pipeline.schema import (
    DROP_CONSTANT_COLUMNS,
    FLOAT_SENSOR_COLUMNS,
    INT_SENSOR_COLUMNS,
    LABEL_COLUMNS,
    SENSOR_COLUMNS,
)


def _make_bronze(n: int = 8, is_test: bool = True) -> pd.DataFrame:
    """Build a minimal Bronze-shaped frame: time + sensors + constants +
    provenance (+ labels)."""
    data: dict[str, list] = {"time": list(range(n))}
    for c in INT_SENSOR_COLUMNS:
        data[c] = [0, 1] * (n // 2)
    for c in FLOAT_SENSOR_COLUMNS:
        data[c] = [float(i) for i in range(n)]
    for c in DROP_CONSTANT_COLUMNS:
        data[c] = [7] * n
    data["source_file"] = ["f.csv.gz"] * n
    data["ingested_at"] = ["2026-01-01T00:00:00+00:00"] * n
    if is_test:
        for c in LABEL_COLUMNS:
            data[c] = [0, 1] * (n // 2)
    else:
        for c in LABEL_COLUMNS:
            data[c] = [0] * n
    return pd.DataFrame(data)


def test_constant_columns_dropped():
    silver = build_silver_frame(_make_bronze(is_test=True), is_test=True)
    for c in DROP_CONSTANT_COLUMNS:
        assert c not in silver.columns


def test_signal_columns_kept():
    silver = build_silver_frame(_make_bronze(is_test=True), is_test=True)
    for c in ("P1_PCV02D", "P2_Emerg", "P2_OnOff"):
        assert c in silver.columns


def test_all_sensor_columns_present():
    silver = build_silver_frame(_make_bronze(is_test=True), is_test=True)
    for c in SENSOR_COLUMNS:
        assert c in silver.columns


def test_train_labels_dropped():
    silver = build_silver_frame(_make_bronze(is_test=False), is_test=False)
    for c in LABEL_COLUMNS:
        assert c not in silver.columns


def test_test_labels_kept():
    silver = build_silver_frame(_make_bronze(is_test=True), is_test=True)
    for c in LABEL_COLUMNS:
        assert c in silver.columns


def test_schema_rejects_out_of_domain_label():
    bad = _make_bronze(is_test=True)
    bad.loc[0, "attack"] = 2
    with pytest.raises(ValueError, match="Silver validation failed"):
        build_silver_frame(bad, is_test=True)


def test_fingerprint_stable_and_sensitive():
    df1 = _make_bronze(is_test=True)
    df2 = _make_bronze(is_test=True)
    cols = list(SENSOR_COLUMNS)
    assert fingerprint(df1, cols) == fingerprint(df2, cols)
    df2.loc[0, FLOAT_SENSOR_COLUMNS[0]] = 999.0
    assert fingerprint(df1, cols) != fingerprint(df2, cols)
