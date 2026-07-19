"""Pandera schema for the HAI Silver layer.

Silver is Bronze with the 19 always-constant columns dropped and dtypes
validated. Three columns that are constant in train but vary under attack
(P1_PCV02D, P2_Emerg, P2_OnOff) are KEPT — they are high-value discrete signals.

Column lists are derived once here so downstream stages import them instead of
re-deriving. Validation uses lazy=True so every violation is reported at once;
callers catch both SchemaError and SchemaErrors.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

# --- Columns dropped from Bronze -> Silver (constant in BOTH train and test) ---
DROP_CONSTANT_COLUMNS: tuple[str, ...] = (
    "P1_PP01AD",
    "P1_PP01AR",
    "P1_PP01BD",
    "P1_PP01BR",
    "P1_PP02D",
    "P1_PP02R",
    "P1_STSP",
    "P2_ASD",
    "P2_AutoGO",
    "P2_MSD",
    "P2_ManualGO",
    "P2_RTR",
    "P2_TripEx",
    "P2_VTR01",
    "P2_VTR02",
    "P2_VTR03",
    "P2_VTR04",
    "P3_LH",
    "P3_LL",
)

# --- Discrete/integer actuator columns kept in Silver ---
# Includes the 3 attack-signal columns (constant in train, vary in test).
INT_SENSOR_COLUMNS: tuple[str, ...] = (
    "P1_PCV02D",
    "P2_Emerg",
    "P2_OnOff",
    "P4_HT_PS",
    "P4_ST_PS",
)

# --- Continuous float sensor columns kept in Silver ---
FLOAT_SENSOR_COLUMNS: tuple[str, ...] = (
    "P1_B2004",
    "P1_B2016",
    "P1_B3004",
    "P1_B3005",
    "P1_B4002",
    "P1_B4005",
    "P1_B400B",
    "P1_B4022",
    "P1_FCV01D",
    "P1_FCV01Z",
    "P1_FCV02D",
    "P1_FCV02Z",
    "P1_FCV03D",
    "P1_FCV03Z",
    "P1_FT01",
    "P1_FT01Z",
    "P1_FT02",
    "P1_FT02Z",
    "P1_FT03",
    "P1_FT03Z",
    "P1_LCV01D",
    "P1_LCV01Z",
    "P1_LIT01",
    "P1_PCV01D",
    "P1_PCV01Z",
    "P1_PCV02Z",
    "P1_PIT01",
    "P1_PIT02",
    "P1_TIT01",
    "P1_TIT02",
    "P2_24Vdc",
    "P2_CO_rpm",
    "P2_HILout",
    "P2_SIT01",
    "P2_SIT02",
    "P2_VT01",
    "P2_VXT02",
    "P2_VXT03",
    "P2_VYT02",
    "P2_VYT03",
    "P3_FIT01",
    "P3_LCP01D",
    "P3_LCV01D",
    "P3_LIT01",
    "P3_PIT01",
    "P4_HT_FD",
    "P4_HT_LD",
    "P4_HT_PO",
    "P4_LD",
    "P4_ST_FD",
    "P4_ST_GOV",
    "P4_ST_LD",
    "P4_ST_PO",
    "P4_ST_PT01",
    "P4_ST_TT01",
)

# All modelling feature columns (order: int then float).
SENSOR_COLUMNS: tuple[str, ...] = INT_SENSOR_COLUMNS + FLOAT_SENSOR_COLUMNS

# --- Label columns (present only in test/attack data) ---
LABEL_COLUMNS: tuple[str, ...] = ("attack", "attack_P1", "attack_P2", "attack_P3")

TIME_COLUMN = "time"
PROVENANCE_COLUMNS: tuple[str, ...] = ("source_file", "ingested_at")


def _sensor_columns_spec() -> dict[str, Column]:
    """Build the Column spec for every sensor column with the right dtype."""
    spec: dict[str, Column] = {}
    for name in INT_SENSOR_COLUMNS:
        spec[name] = Column(int, nullable=False, coerce=True)
    for name in FLOAT_SENSOR_COLUMNS:
        spec[name] = Column(float, nullable=False, coerce=True)
    return spec


def _label_columns_spec() -> dict[str, Column]:
    """Labels are 0/1 integers."""
    return {
        name: Column(int, pa.Check.isin([0, 1]), nullable=False, coerce=True)
        for name in LABEL_COLUMNS
    }


# Schema for train Silver: sensors + provenance, NO labels.
SILVER_TRAIN_SCHEMA = DataFrameSchema(
    {
        **_sensor_columns_spec(),
        "source_file": Column(str, nullable=False),
        "ingested_at": Column(str, nullable=False),
    },
    strict=False,  # allow the time column through without a per-value check
    coerce=True,
)

# Schema for test Silver: sensors + provenance + labels.
SILVER_TEST_SCHEMA = DataFrameSchema(
    {
        **_sensor_columns_spec(),
        **_label_columns_spec(),
        "source_file": Column(str, nullable=False),
        "ingested_at": Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)

# --- Gold split assignment (see EDA.md: chosen so both calib and holdout
#     retain all three attack types, especially the rare P3) ---
GOLD_SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("train1", "train2", "train3"),
    "test_calib": ("test2", "test4"),
    "test_holdout": ("test1", "test3", "test5"),
}
