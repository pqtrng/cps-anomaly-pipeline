"""Tests for T5 evaluation metrics.

These pin down the properties T5 is judged on, especially the honesty logic:
  1. the threshold is a percentile of NORMAL calib windows only — attacks in
     calib do not pull it up;
  2. raw window metrics match hand-computed confusion counts;
  3. point-adjust inflates recall (a segment with one flagged window counts as
     fully detected) yet cannot conjure detection of an all-missed segment;
  4. per-process recall counts only that process's attack windows, P4 absent;
  5. attack intervals segment row labels correctly;
  6. interval detection latency = first alert time minus onset, and a missed
     interval reports detected=False / latency None;
  7. SLO attainment counts undetected attacks as misses;
  8. the one-detector bundle reports the raw-vs-PA recall delta.

All synthetic, all pure arrays — no model, no disk.
"""

from __future__ import annotations

import numpy as np

from cps_anomaly_pipeline.evaluate import (
    attack_intervals,
    binary_metrics,
    evaluate_detector,
    fit_threshold_calib_normal,
    interval_detections,
    per_process_recall,
    point_adjust_metrics,
    summarize_detections,
)


def test_threshold_uses_calib_normal_only():
    """Attack-window scores in calib must not raise the threshold."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=1000)  # label 0
    attacks = np.full(50, 1000.0)  # label 1, huge scores
    scores = np.concatenate([normal, attacks])
    labels = np.concatenate([np.zeros(1000, int), np.ones(50, int)])

    thr = fit_threshold_calib_normal(scores, labels, percentile=99.5)
    # Threshold must sit in the normal tail, nowhere near the 1000-valued attacks.
    assert thr == np.percentile(normal, 99.5)
    assert thr < 10.0


def test_binary_metrics_known_confusion():
    #        idx: 0  1  2  3  4
    flags = np.array([1, 1, 0, 0, 1], dtype=bool)
    labels = np.array([1, 0, 1, 0, 1], dtype=int)
    m = binary_metrics(flags, labels)
    # tp: idx0,idx4 = 2 ; fp: idx1 = 1 ; fn: idx2 = 1 ; tn: idx3 = 1
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 1)
    assert m.precision == 2 / 3
    assert m.recall == 2 / 3
    assert abs(m.f1 - (2 / 3)) < 1e-12


def test_point_adjust_inflates_recall():
    """A true segment with a single flagged window scores full recall under PA."""
    # windows 2..6 are one attack segment; only window 4 is flagged.
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=int)
    flags = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)

    raw = binary_metrics(flags, labels)
    pa = point_adjust_metrics(flags, labels)
    # Raw: 1 of 5 attack windows caught.
    assert raw.recall == 1 / 5
    # Point-adjust: the whole segment is marked detected -> recall 1.0.
    assert pa.recall == 1.0
    assert pa.recall > raw.recall


def test_point_adjust_cannot_detect_missed_segment():
    """A segment with zero flagged windows stays undetected under PA."""
    labels = np.array([0, 1, 1, 1, 0], dtype=int)
    flags = np.array([0, 0, 0, 0, 0], dtype=bool)
    pa = point_adjust_metrics(flags, labels)
    assert pa.recall == 0.0
    assert pa.tp == 0


def test_per_process_recall_counts_only_that_process():
    flags = np.array([1, 0, 1, 0, 1], dtype=bool)
    process_labels = {
        "attack": np.array([1, 1, 1, 0, 1], dtype=int),  # skipped by design
        "attack_P1": np.array([1, 1, 0, 0, 0], dtype=int),  # windows 0,1 ; caught 0
        "attack_P2": np.array([0, 0, 1, 0, 1], dtype=int),  # windows 2,4 ; caught both
    }
    rec = per_process_recall(flags, process_labels)
    assert "attack" not in rec
    assert rec["attack_P1"] == 1 / 2  # only window 0 flagged of {0,1}
    assert rec["attack_P2"] == 1.0  # windows 2 and 4 both flagged


def test_attack_intervals_segmentation():
    row_attack = np.array([0, 1, 1, 0, 0, 1, 0], dtype=int)
    assert attack_intervals(row_attack) == [(1, 2), (5, 5)]


def test_interval_detection_latency_and_miss():
    window = 3
    # Two attack intervals in row space.
    intervals = [(10, 20), (50, 60)]
    # Window starts (stride 1) — use a coarse synthetic grid covering both.
    starts = np.arange(0, 70)
    flags = np.zeros(len(starts), dtype=bool)
    # Flag one window that overlaps the FIRST interval. Its start = 9 -> covers
    # rows [9,11], overlaps [10,20]; alert time = 9 + 3 - 1 = 11 ; latency 11-10=1.
    flags[9] = True
    # Second interval left entirely unflagged -> missed.
    dets = interval_detections(intervals, flags, starts, window)
    assert dets[0].detected is True
    assert dets[0].latency_s == 1.0
    assert dets[1].detected is False
    assert dets[1].latency_s is None


def test_slo_attainment_counts_misses():
    window = 3
    intervals = [(10, 20), (50, 60)]
    starts = np.arange(0, 70)
    flags = np.zeros(len(starts), dtype=bool)
    flags[9] = True  # detects interval 1 with latency 1s; interval 2 missed
    summary = summarize_detections(
        interval_detections(intervals, flags, starts, window)
    )
    assert summary.n_attacks == 2
    assert summary.n_detected == 1
    assert summary.detection_rate == 0.5
    # 1 of 2 attacks within 5s (the other is never detected).
    assert summary.slo_attainment(5.0) == 0.5
    assert summary.slo_attainment(0.5) == 0.0  # latency 1s exceeds 0.5s target


def test_evaluate_detector_bundle_reports_delta():
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=int)
    scores = np.array([0, 0, 0, 0, 9, 0, 0, 0, 0], dtype=float)  # only window 4 high
    holdout_labels = {"attack": labels}
    row_attack = labels  # reuse as a row-space stand-in for the bundle test
    starts = np.arange(len(labels))
    out = evaluate_detector(
        holdout_scores=scores,
        holdout_labels=holdout_labels,
        holdout_row_attack=row_attack,
        starts=starts,
        window=1,
        threshold=1.0,
    )
    d = out["point_adjust_inflation"]
    assert d["recall_raw"] == 1 / 5
    assert d["recall_point_adjust"] == 1.0
    assert round(d["recall_delta"], 6) == round(1.0 - 1 / 5, 6)
