"""Content fingerprinting shared across pipeline layers.

A SHA-1 hash over selected columns' values, used to detect whether the
underlying data changed between runs (drift detection, no auto-heal).
"""

from __future__ import annotations

import hashlib

import pandas as pd


def fingerprint(df: pd.DataFrame, cols: list[str]) -> str:
    """SHA-1 over the given columns' values — a stable content fingerprint.

    Columns are sorted so the hash is independent of column order.
    """
    ordered = df[sorted(cols)]
    payload = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    return hashlib.sha1(payload).hexdigest()
