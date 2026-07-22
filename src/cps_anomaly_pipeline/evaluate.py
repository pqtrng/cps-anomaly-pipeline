"""T5 evaluation — pure metrics over per-window scores and labels.

This module holds the evaluation-honesty logic that is the point of T5. Given
per-window scores and per-window labels (from scoring.py, but passed as plain
arrays so this module stays pure — no torch, no pandas, no disk), it computes:

  1. a threshold fit ONLY on the NORMAL windows of the calibration split
     (label == 0), at a percentile — never touching holdout labels;
  2. raw window-level precision / recall / F1;
  3. point-adjusted precision / recall / F1, plus the delta versus raw. The
     point-adjust convention marks an entire true-anomaly segment as detected if
     ANY single window inside it is flagged. It is widespread in the anomaly-
     detection literature and systematically INFLATES recall — reporting both
     side by side, and the inflation delta, is the honest-evaluation point;
  4. per-process recall (P1 / P2 / P3 — HAI has no P4 label, so it is never
     reported);
  5. attack-interval detection and detection latency, framed as an SLI (time
     from attack onset to first alert) with SLO attainment (fraction of attacks
     detected within a target number of seconds).

Row time is HAI's 1 Hz sampling: one row = one second, so a latency in rows is a
latency in seconds. A window covering rows [s, s+window) can only raise an alert
once its last row is observed, so its alert time is ``s + window - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _safe_div(num: float, den: float) -> float:
    """Divide, returning 0.0 when the denominator is 0 (no samples of that kind)."""
    return float(num) / float(den) if den else 0.0


# --- Thresholding -----------------------------------------------------------


def fit_threshold_calib_normal(
    calib_scores: np.ndarray, calib_labels: np.ndarray, percentile: float = 99.5
) -> float:
    """Threshold = a percentile of scores over the NORMAL calib windows only.

    The calibration split may contain attacks; only its normal windows
    (label == 0) describe the model's normal-operation error, so the threshold
    is a percentile of those. Learned from calib alone and applied unchanged to
    holdout — no holdout window influences its own flag. p99.5 mirrors the T3
    train-fit convention (~0.5% target false-positive rate on normal).
    """
    normal = calib_scores[calib_labels == 0]
    if normal.size == 0:
        raise ValueError("no normal (label==0) windows in calib to fit a threshold")
    return float(np.percentile(normal, percentile))


def flags_from_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean flags: True where score strictly exceeds the threshold."""
    return scores > threshold


# --- Window-level metrics ---------------------------------------------------


@dataclass
class ClassMetrics:
    """Binary detection metrics with the confusion counts they came from."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_positive: int  # windows whose label is 1
    n_flagged: int  # windows flagged anomalous

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "n_positive": self.n_positive,
            "n_flagged": self.n_flagged,
        }


def binary_metrics(flags: np.ndarray, labels: np.ndarray) -> ClassMetrics:
    """Precision / recall / F1 from flags vs 0/1 window labels."""
    flags = flags.astype(bool)
    truth = labels.astype(bool)
    tp = int(np.sum(flags & truth))
    fp = int(np.sum(flags & ~truth))
    fn = int(np.sum(~flags & truth))
    tn = int(np.sum(~flags & ~truth))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return ClassMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_positive=int(truth.sum()),
        n_flagged=int(flags.sum()),
    )


# --- Point-adjust (the honesty centrepiece) ---------------------------------


def _segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of label == 1 as inclusive (start, end) index pairs."""
    truth = labels.astype(bool)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(truth):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(truth) - 1))
    return segments


def point_adjust_flags(flags: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Apply the point-adjust convention to the flags.

    For each contiguous true-anomaly segment, if ANY window in it is flagged,
    mark EVERY window in that segment as flagged. Flags outside true segments
    are left untouched, so false positives are not adjusted away — only the
    recall side is inflated, which is exactly the effect T5 exposes.
    """
    adjusted = flags.astype(bool).copy()
    for a, b in _segments(labels):
        if adjusted[a : b + 1].any():
            adjusted[a : b + 1] = True
    return adjusted


def point_adjust_metrics(flags: np.ndarray, labels: np.ndarray) -> ClassMetrics:
    """Metrics after applying the point-adjust convention. Compare to raw."""
    return binary_metrics(point_adjust_flags(flags, labels), labels)


# --- Per-process recall -----------------------------------------------------


def per_process_recall(
    flags: np.ndarray, process_labels: dict[str, np.ndarray]
) -> dict[str, float]:
    """Recall on the windows carrying each per-process attack label (==1).

    Recall = fraction of that process's attack windows that were flagged. The
    top-level ``attack`` column is skipped here (it is covered by the window
    metrics above); only per-process columns are reported, and only those the
    split actually carries (HAI has no P4, so P4 never appears).
    """
    out: dict[str, float] = {}
    for name, plabels in process_labels.items():
        if name == "attack":
            continue
        truth = plabels.astype(bool)
        detected = int(np.sum(flags.astype(bool) & truth))
        out[name] = _safe_div(detected, int(truth.sum()))
    return out


# --- Attack-interval detection and latency (SLI / SLO) ----------------------


def attack_intervals(row_attack: np.ndarray) -> list[tuple[int, int]]:
    """Attack intervals in ROW space as inclusive (onset_row, end_row) pairs.

    Row-level (not window-level) because latency is measured against the true
    attack onset in time. Each contiguous run of attacked rows is one attack.
    """
    return _segments(row_attack)


@dataclass
class IntervalDetection:
    """Whether one attack interval was detected, and how late (seconds)."""

    onset: int  # attack onset row (== second, HAI is 1 Hz)
    end: int  # attack end row (inclusive)
    detected: bool
    latency_s: float | None  # seconds from onset to first alert; None if missed


def interval_detections(
    intervals: list[tuple[int, int]],
    flags: np.ndarray,
    starts: np.ndarray,
    window: int,
) -> list[IntervalDetection]:
    """For each attack interval, find the first flagged window that overlaps it.

    A window starting at row ``s`` covers rows ``[s, s+window)`` and can raise an
    alert only at its last row, so its alert time is ``s + window - 1``. It
    reports on an interval [a, b] when it overlaps it (``s <= b`` and
    ``s+window-1 >= a``). Because overlap already requires ``s+window-1 >= a``,
    the alert time is always at or after onset, so latency is never negative.
    Latency is the earliest such alert time minus the onset, in seconds.
    """
    starts = np.asarray(starts)
    alert_time = starts + window - 1
    flags = flags.astype(bool)
    out: list[IntervalDetection] = []
    for a, b in intervals:
        overlaps = (starts <= b) & (alert_time >= a) & flags
        if overlaps.any():
            first_alert = int(alert_time[overlaps].min())
            out.append(
                IntervalDetection(
                    onset=a, end=b, detected=True, latency_s=float(first_alert - a)
                )
            )
        else:
            out.append(
                IntervalDetection(onset=a, end=b, detected=False, latency_s=None)
            )
    return out


@dataclass
class DetectionSummary:
    """Event-level (per-attack) detection summary with latency and SLO stats."""

    n_attacks: int
    n_detected: int
    detection_rate: float  # per-attack recall (event-level)
    latencies_s: list[float] = field(default_factory=list)  # detected only

    @property
    def median_latency_s(self) -> float | None:
        return float(np.median(self.latencies_s)) if self.latencies_s else None

    @property
    def mean_latency_s(self) -> float | None:
        return float(np.mean(self.latencies_s)) if self.latencies_s else None

    @property
    def max_latency_s(self) -> float | None:
        return float(np.max(self.latencies_s)) if self.latencies_s else None

    def slo_attainment(self, target_s: float) -> float:
        """Fraction of ALL attacks detected within ``target_s`` seconds of onset.

        Denominator is every attack (undetected counts as a miss), so this reads
        as an operational SLO: "share of attacks alerted within the target".
        """
        if self.n_attacks == 0:
            return 0.0
        within = sum(1 for lat in self.latencies_s if lat <= target_s)
        return within / self.n_attacks

    def to_dict(self, slo_targets_s: tuple[float, ...] = (10, 30, 60)) -> dict:
        return {
            "n_attacks": self.n_attacks,
            "n_detected": self.n_detected,
            "detection_rate": round(self.detection_rate, 6),
            "median_latency_s": self.median_latency_s,
            "mean_latency_s": self.mean_latency_s,
            "max_latency_s": self.max_latency_s,
            "slo_attainment": {
                f"within_{int(t)}s": round(self.slo_attainment(t), 6)
                for t in slo_targets_s
            },
        }


def summarize_detections(dets: list[IntervalDetection]) -> DetectionSummary:
    """Roll interval detections up into a per-attack detection summary."""
    latencies = [d.latency_s for d in dets if d.detected and d.latency_s is not None]
    n_detected = sum(1 for d in dets if d.detected)
    return DetectionSummary(
        n_attacks=len(dets),
        n_detected=n_detected,
        detection_rate=_safe_div(n_detected, len(dets)),
        latencies_s=latencies,
    )


# --- One-detector bundle (consumed by the T5 runner, File 3) ----------------


def evaluate_detector(
    holdout_scores: np.ndarray,
    holdout_labels: dict[str, np.ndarray],
    holdout_row_attack: np.ndarray,
    starts: np.ndarray,
    window: int,
    threshold: float,
    slo_targets_s: tuple[float, ...] = (10, 30, 60),
) -> dict:
    """Full honest report for ONE detector on the holdout, at a fixed threshold.

    Bundles raw vs point-adjust window metrics (with the recall inflation delta),
    per-process recall, and event-level detection + latency/SLO. The threshold is
    supplied by the caller (fit on calib-normal), never derived from holdout.
    """
    flags = flags_from_threshold(holdout_scores, threshold)
    raw = binary_metrics(flags, holdout_labels["attack"])
    pa = point_adjust_metrics(flags, holdout_labels["attack"])
    dets = interval_detections(
        attack_intervals(holdout_row_attack), flags, starts, window
    )
    summary = summarize_detections(dets)
    return {
        "threshold": round(float(threshold), 6),
        "window_metrics_raw": raw.to_dict(),
        "window_metrics_point_adjust": pa.to_dict(),
        "point_adjust_inflation": {
            "recall_raw": round(raw.recall, 6),
            "recall_point_adjust": round(pa.recall, 6),
            "recall_delta": round(pa.recall - raw.recall, 6),
            "f1_raw": round(raw.f1, 6),
            "f1_point_adjust": round(pa.f1, 6),
            "f1_delta": round(pa.f1 - raw.f1, 6),
        },
        "per_process_recall": {
            k: round(v, 6) for k, v in per_process_recall(flags, holdout_labels).items()
        },
        "detection": summary.to_dict(slo_targets_s),
    }
