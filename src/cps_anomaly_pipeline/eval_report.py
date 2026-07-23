"""T5 runner — the honest evaluation report comparing both detectors.

Wires scoring.py and evaluate.py into one command (``cps-eval``):

  1. resolve a training run (latest under ``runs/lstm_ae/`` or ``--run-dir``)
     and load its ``best.pt`` + ``int_scaler.npz``;
  2. fit the z-score baseline on Gold train (as T3 defined it);
  3. score test_calib and test_holdout with BOTH detectors on the shared
     stride-1 window grid;
  4. for each detector stream, fit a threshold on the calib NORMAL windows and
     apply it unchanged to holdout — holdout labels never touched at fit time;
  5. compute the full honest report per detector (raw vs point-adjust window
     metrics with the recall-inflation delta, per-process recall, event-level
     detection rate + latency SLI / SLO), plus a train-normal threshold
     cross-check;
  6. print a side-by-side comparison table and write the canonical, versioned
     ``results/eval_metrics.json``.

Three detector streams are compared on the same holdout: the LSTM-AE and the two
z-score baselines (max|z| and mean z²). All numbers in the results file are the
canonical T5 figures — produced from real Gold + a real checkpoint on the
machine that has them, then committed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cps_anomaly_pipeline.baseline import SCORING_METHODS, BaselineModel
from cps_anomaly_pipeline.device import get_device
from cps_anomaly_pipeline.evaluate import (
    evaluate_detector,
    fit_threshold_calib_normal,
)
from cps_anomaly_pipeline.paths import PathConfig
from cps_anomaly_pipeline.schema import LABEL_COLUMNS
from cps_anomaly_pipeline.scoring import (
    EVAL_STRIDE,
    load_int_scaler,
    load_lstm_ae,
    score_ae,
    score_baseline,
    score_split,
)
from cps_anomaly_pipeline.windowing import DEFAULT_WINDOW

DEFAULT_PERCENTILE = 99.5
DEFAULT_MODEL = "lstm_ae"
RESULTS_FILENAME = "eval_metrics.json"

# Detector stream labels in the order they appear in the table / results file.
DETECTOR_ORDER = ("lstm_ae", "baseline_max_abs_z", "baseline_mean_sq_z")


def resolve_run_dir(
    runs_root: Path, model: str = DEFAULT_MODEL, run_dir: str | Path | None = None
) -> Path:
    """Return the run directory to evaluate: explicit override or latest run.

    Latest is the lexicographically last timestamped dir under runs/<model>/,
    matching the Makefile ``save-metrics`` convention (run ids are timestamps,
    so lexicographic == chronological).
    """
    if run_dir is not None:
        path = Path(run_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"--run-dir not a directory: {path}")
        return path
    model_root = runs_root / model
    candidates = sorted(p for p in model_root.glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no runs under {model_root}; train first")
    return candidates[-1]


def load_gold_splits(gold_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the three Gold splits written by the Gold stage."""
    names = ("train", "test_calib", "test_holdout")
    return {n: pd.read_parquet(gold_dir / f"{n}.parquet") for n in names}


def _train_crosscheck_thresholds(
    model,
    int_scaler,
    baseline: BaselineModel,
    train_df: pd.DataFrame,
    percentile: float,
    window: int,
    device: str | None = None,
) -> dict[str, float]:
    """Train-normal threshold per detector, for a sanity cross-check only.

    Train is entirely normal, so no label filtering is needed. Scored on
    NON-OVERLAPPING windows (stride = window) purely to keep this extra pass
    cheap — a percentile is insensitive to the stride at this sample size. These
    thresholds are reported beside the operating (calib-normal) thresholds so a
    reader can see the two agree; they are not used to flag anything.
    """
    stride = window  # non-overlapping windows for a cheap, unbiased sample
    ae_scores = score_ae(
        model, int_scaler, train_df, window=window, stride=stride, device=device
    )
    out = {"lstm_ae": float(np.percentile(ae_scores, percentile))}
    for method in SCORING_METHODS:
        scores = score_baseline(
            baseline, train_df, method, window=window, stride=stride
        )
        out[f"baseline_{method}"] = float(np.percentile(scores, percentile))
    return out


def _read_json_if_present(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def run_eval(
    paths: PathConfig | None = None,
    run_dir: str | Path | None = None,
    percentile: float = DEFAULT_PERCENTILE,
    crosscheck: bool = True,
    results_dir: Path | None = None,
    window: int = DEFAULT_WINDOW,
    device: str | None = None,
) -> dict:
    """Run the full T5 evaluation and write the canonical results file.

    ``device`` pins the AE scoring device (e.g. "cpu"); default auto-selects. The
    resolved device is recorded in the results so the numbers are reproducible.

    Returns the results dict (also written to ``<results_dir>/eval_metrics.json``).
    """
    paths = paths or PathConfig()
    results_dir = results_dir or Path("results")
    eval_device = device or get_device()
    run_dir = resolve_run_dir(paths.runs_dir, DEFAULT_MODEL, run_dir)

    model = load_lstm_ae(run_dir / "best.pt")
    int_scaler = load_int_scaler(run_dir / "int_scaler.npz")

    gold = load_gold_splits(paths.hai_gold_dir)
    baseline = BaselineModel.fit(gold["train"])

    calib_ws = score_split(
        model,
        int_scaler,
        baseline,
        gold["test_calib"],
        LABEL_COLUMNS,
        window,
        device=eval_device,
    )
    holdout_ws = score_split(
        model,
        int_scaler,
        baseline,
        gold["test_holdout"],
        LABEL_COLUMNS,
        window,
        device=eval_device,
    )

    # Detector stream -> (calib window scores, holdout window scores).
    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "lstm_ae": (calib_ws.ae, holdout_ws.ae),
        "baseline_max_abs_z": (
            calib_ws.baseline["max_abs_z"],
            holdout_ws.baseline["max_abs_z"],
        ),
        "baseline_mean_sq_z": (
            calib_ws.baseline["mean_sq_z"],
            holdout_ws.baseline["mean_sq_z"],
        ),
    }

    holdout_row_attack = gold["test_holdout"]["attack"].to_numpy(dtype=np.int64)

    crosscheck_thr = (
        _train_crosscheck_thresholds(
            model,
            int_scaler,
            baseline,
            gold["train"],
            percentile,
            window,
            device=eval_device,
        )
        if crosscheck
        else {}
    )

    detectors: dict[str, dict] = {}
    for label in DETECTOR_ORDER:
        calib_scores, holdout_scores = streams[label]
        threshold = fit_threshold_calib_normal(
            calib_scores, calib_ws.labels["attack"], percentile
        )
        report = evaluate_detector(
            holdout_scores=holdout_scores,
            holdout_labels=holdout_ws.labels,
            holdout_row_attack=holdout_row_attack,
            starts=holdout_ws.starts,
            window=window,
            threshold=threshold,
        )
        if crosscheck:
            train_thr = crosscheck_thr[label]
            report["threshold_crosscheck"] = {
                "calib_normal": round(threshold, 6),
                "train_normal": round(train_thr, 6),
                "ratio": round(threshold / train_thr, 6) if train_thr else None,
            }
        detectors[label] = report

    run_metrics = _read_json_if_present(run_dir / "metrics.json")
    manifest = _read_json_if_present(paths.hai_gold_dir / "gold_manifest.json")

    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": f"{run_dir.parent.name}/{run_dir.name}",
        "eval_device": eval_device,
        "window": window,
        "stride": EVAL_STRIDE,
        "threshold_percentile": percentile,
        "checkpoint_best_val_loss": (
            run_metrics.get("best_val_loss") if run_metrics else None
        ),
        "gold_splits": (manifest.get("splits") if manifest else None),
        "detectors": detectors,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / RESULTS_FILENAME).write_text(json.dumps(results, indent=2))
    return results


# --- Console table ----------------------------------------------------------


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return format(value, spec) if value is not None else "  -  "


def format_table(results: dict) -> str:
    """Compact side-by-side comparison of the detectors on the holdout."""
    header = (
        f"{'detector':<20} {'P':>6} {'R':>6} {'F1':>6} "
        f"{'PA_R':>6} {'det%':>6} {'medLat':>7} {'SLO30':>6}"
    )
    lines = [header, "-" * len(header)]
    for label in DETECTOR_ORDER:
        d = results["detectors"][label]
        raw = d["window_metrics_raw"]
        pa = d["window_metrics_point_adjust"]
        det = d["detection"]
        lines.append(
            f"{label:<20} "
            f"{_fmt(raw['precision'])} {_fmt(raw['recall'])} {_fmt(raw['f1'])} "
            f"{_fmt(pa['recall'])} "
            f"{_fmt(det['detection_rate'])} "
            f"{_fmt(det['median_latency_s'], '.0f'):>7} "
            f"{_fmt(det['slo_attainment']['within_30s']):>6}"
        )
    lines.append("")
    lines.append("P/R/F1 = raw window-level; PA_R = point-adjust recall (inflated);")
    lines.append("det% = per-attack detection rate; medLat = median latency (s);")
    lines.append("SLO30 = fraction of attacks alerted within 30 s.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="T5 honest evaluation report.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="run directory with best.pt + int_scaler.npz (default: latest)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="calib-normal threshold percentile (default: 99.5)",
    )
    parser.add_argument(
        "--no-crosscheck",
        action="store_true",
        help="skip the train-normal threshold cross-check (faster)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="pin AE scoring device, e.g. 'cpu' for a CI-reproducible number "
        "(default: auto-select)",
    )
    args = parser.parse_args()

    paths = PathConfig()
    print(f"Reading Gold from: {paths.hai_gold_dir}")
    results = run_eval(
        paths=paths,
        run_dir=args.run_dir,
        percentile=args.percentile,
        crosscheck=not args.no_crosscheck,
        device=args.device,
    )
    print(f"Run: {results['run_dir']}  (eval on {results['eval_device']})")
    print(
        f"Threshold: calib-normal p{results['threshold_percentile']} "
        f"| window={results['window']} stride={results['stride']}\n"
    )
    print(format_table(results))
    print(f"\nWrote results/{RESULTS_FILENAME}")


if __name__ == "__main__":
    main()
