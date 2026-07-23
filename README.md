[![CI](https://github.com/pqtrng/cps-anomaly-pipeline/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/pqtrng/cps-anomaly-pipeline/actions/workflows/ci.yml)

# cps-anomaly-pipeline

A DE-first anomaly-detection pipeline for cyber-physical systems, built on the
[HAI](https://github.com/icsdataset/hai) industrial control-system testbed. The emphasis is production
data infrastructure — a Medallion-layered pipeline, schema validation, reproducible runs, and **honest
evaluation** — rather than a single model in a notebook. The model is the smallest part of the story;
the leakage discipline, the calibration split, and the raw-vs-inflated metric reporting are the point.

## Status

**T6 complete** — production docs with SLO/SLI framing and honest, calibration-based evaluation, on top
of the full Bronze → Silver → Gold → detector → evaluation pipeline.

| Stage | Description                                               | State |
| ----- | --------------------------------------------------------- | ----- |
| T1    | Ingest HAI `.csv.gz` → Bronze (Parquet, raw-as-ingested)  | ✅     |
| T2    | Silver / Gold + Pandera validation + SHA-1 fingerprinting | ✅     |
| T3    | Baseline detector (z-score, both scoring methods)         | ✅     |
| T4    | LSTM-Autoencoder (windowing, TensorBoard, val-loss ckpt)  | ✅     |
| T5    | Honest evaluation (per-attack, detection lead time)       | ✅     |
| T6    | SLO / SLI framing + production docs (IEC 62443)           | ✅     |
| T7    | Tests + CI                                                | ✅     |

## Where this sits — the OT monitoring gap

Cybersecurity for semiconductor-fab **equipment** is specified by **SEMI E187**, which scopes the
equipment computing environment (endpoint OS hardening, network configuration, update management). E187
deliberately leaves the **PLC / SCADA process-control layer beneath the equipment out of scope.**

The HAI testbed operates at exactly that excluded layer — boiler, turbine, and water-treatment processes
driven by PLCs. So this pipeline does **not** "map to SEMI E187." It addresses the **OT process-control
monitoring gap that E187 leaves out of scope.** The applicable framework for that layer is
**IEC 62443** (security for industrial automation and control systems), under which continuous monitoring
and anomaly detection are recognised operational controls. This project is a monitoring building block
for that layer — not a claim of fab-equipment compliance.

## Pipeline architecture

Medallion layering, with every scaler and threshold fit on a training split and applied unchanged
downstream — the leakage boundary is the design, not an afterthought.

```text
raw HAI .csv.gz
      │  cps-ingest
      ▼
[ Bronze ]  raw-as-ingested Parquet + provenance (source_file, ingested_at)
      │  cps-silver   drop 19 constant cols; keep 3 attack-signal cols
      ▼                (P1_PCV02D / P2_Emerg / P2_OnOff); Pandera + SHA-1
[ Silver ]  cleaned, typed (float sensors vs int actuators), validated
      │  cps-gold     StandardScaler on 55 float sensors  (fit TRAIN only)
      ▼                int features min-max scaled        (fit TRAIN only)
[ Gold ]    file-based splits + manifest:
      │        train        = train1-3   (normal only, 921,603 rows)
      │        test_calib   = test2,4    (threshold calibration, 158,402 rows)
      │        test_holdout = test1,3,5  (report numbers here, 243,603 rows)
      │
      ├─ cps-baseline   z-score detector (max|z| and mean z²), per-row
      ├─ cps-train      LSTM-AE on normal windows (window=60, TensorBoard, val-loss ckpt)
      │
      ▼  cps-eval
[ Evaluation ]  per-window scoring on a shared stride-1 grid;
                threshold fit on CALIB-normal windows, applied unchanged to holdout;
                raw + point-adjust metrics, per-process recall, detection latency SLI/SLO
                → results/eval_metrics.json
```

**Leakage prevention (non-negotiable):** the feature scaler is fit on train-normal only; the detection
threshold is fit on **calib-normal windows** only and applied unchanged to holdout; splits are by file so
no window crosses a file boundary. Holdout numbers are never used to tune anything.

## Detectors

- **z-score baseline** — per-feature standardisation, two scoring methods: `max|z|` (worst single feature
  in a row) and `mean z²` (average squared deviation). Cheap, self-normalising, and a genuinely
  competitive reference — the LSTM-AE has to beat it to justify its existence.
- **LSTM-Autoencoder** — trained on normal-only windows (window=60, 60 features = 55 float + 5 int,
  hidden=64, 1 layer, fixed LR 1e-3, seed 42). Best checkpoint by **val loss** (0.08301 @ epoch 298).
  Anomaly score = per-window reconstruction MSE. `ReduceLROnPlateau` was tried and discarded — on HAI the
  loss descends steadily rather than plateauing, so it gave a worse val loss (documented in `train.py`).

## Evaluation methodology

Two honesty rules shape every number below.

**1. Raw metrics are primary; point-adjust is reported only alongside, with its inflation made explicit.**
Point-adjust (PA) marks an entire attack interval as detected if *any single window* inside it fires. It
is common in ICS-anomaly papers and it inflates recall toward 1.0 while hiding how much of each attack is
actually flagged. We report raw window-level precision/recall/F1 as the headline, and PA next to it with
the `recall_delta` / `f1_delta` spelled out. If you only read one column, read the raw one.

**2. The threshold is calibrated on a held-out calib-normal split, not on train.** See the crosscheck
below — a train-fit threshold would be too tight for the shifted test-normal distribution and would
inflate false alarms.

### SLO / SLI framing

Treating detection as an operational service:

- **SLI (indicator, measured):** *detection latency* — seconds from an attack interval's onset to the
  first alerting window. HAI samples at 1 Hz, so one row ≈ one second. Alert time of a window starting at
  row `s` is `s + window − 1`; latency is clamped to ≥ 0.
- **SLO (objective, target):** *detect the attack within 30 s of onset.* `slo_attainment.within_30s` is the
  fraction of the 25 holdout attack intervals alerted within that budget. 10 s and 60 s budgets are
  reported for shape.

This reframes "did it detect the attack" (a binary) into "how fast, on what fraction of attacks" — the
question an operator actually cares about.

## Results (canonical — `results/eval_metrics.json`, `eval_device=cpu`)

Holdout = test1,3,5 (243,603 rows, 25 attack intervals). Numbers are reproducible: canonical eval pins
`--device cpu` to match CI.

### Detection quality — window level, **raw** (headline)

| Detector          | Precision | Recall | F1    |
| ----------------- | --------- | ------ | ----- |
| **LSTM-AE**       | 0.814     | 0.772  | 0.792 |
| baseline max\|z\| | 0.722     | 0.695  | 0.708 |
| baseline mean z²  | 0.736     | 0.741  | 0.738 |

The LSTM-AE leads on every raw figure, but the margin over `mean z²` is modest — the honest read is that a
well-calibrated z-score baseline is hard to beat on this data.

### Raw vs point-adjust — the inflation callout

| Detector          | Recall raw → PA        | F1 raw → PA            |
| ----------------- | ---------------------- | ---------------------- |
| **LSTM-AE**       | 0.772 → 1.000 (+0.228) | 0.792 → 0.919 (+0.127) |
| baseline max\|z\| | 0.695 → 0.948 (+0.253) | 0.708 → 0.856 (+0.148) |
| baseline mean z²  | 0.741 → 0.948 (+0.207) | 0.738 → 0.857 (+0.148) |

PA lifts LSTM-AE recall to a perfect 1.000 — which is precisely why it is *not* the headline. Reporting PA
alone would overstate every detector by 20–25 recall points.

### Per-process recall — **raw** (P4 unlabelled in HAI)

| Detector          | P1    | P2    | P3    |
| ----------------- | ----- | ----- | ----- |
| **LSTM-AE**       | 0.754 | 0.892 | 0.814 |
| baseline max\|z\| | 0.667 | 0.880 | 0.594 |
| baseline mean z²  | 0.720 | 0.882 | 0.737 |

HAI 21.03 does not label process stage 4, so no P4 recall is reportable even though P4 sensors feed the
model. P1 is the weakest stage for every detector — an open question (see below).

### Detection latency — SLI / SLO

| Detector          | Detected     | Median | Mean   | Max   | SLO@10s | SLO@30s  | SLO@60s |
| ----------------- | ------------ | ------ | ------ | ----- | ------- | -------- | ------- |
| **LSTM-AE**       | 25/25 (1.00) | 4.0 s  | 17.6 s | 164 s | 0.56    | **0.88** | 0.96    |
| baseline max\|z\| | 24/25 (0.96) | 14.5 s | 20.9 s | 84 s  | 0.44    | 0.72     | 0.84    |
| baseline mean z²  | 24/25 (0.96) | 9.5 s  | 16.3 s | 83 s  | 0.48    | 0.80     | 0.88    |

LSTM-AE detects **all** 25 attacks (baselines miss one each) and hits the 30 s SLO on 88% of them, with a
4 s median. Its 164 s max is the one genuinely hard case discussed below.

### Threshold crosscheck (distribution shift)

| Detector          | calib-normal thr | train-normal thr | ratio     |
| ----------------- | ---------------- | ---------------- | --------- |
| **LSTM-AE**       | 0.2894           | 0.1768           | **1.637** |
| baseline max\|z\| | 13.117           | 13.124           | 0.999     |
| baseline mean z²  | 5.570            | 5.454            | 1.021     |

False-alarm rate on holdout-normal (LSTM-AE): **0.43%** (1,027 / 237,728 normal windows).

## Limitations & Open Questions

This is a learner's project and the failure modes are as interesting as the wins.

- **(a) Tail latency — LSTM-AE max = 164 s.** One slow attack drives the tail: it mimics normal dynamics,
  so reconstruction error stays under threshold until the deviation accumulates. Median (4 s) is fine; the
  tail is not, and for an SLO the tail is what bites. Related to (e).
- **(b) Point-adjust is a trap.** It reports 1.000 recall for LSTM-AE by crediting a whole interval for a
  single fired window. We keep it *only* beside the raw number with the delta shown. Treat any ICS paper
  quoting PA-only recall with suspicion.
- **(c) Crosscheck ratio 1.64 = distribution shift.** The LSTM-AE threshold calibrated on calib-normal is
  1.64× the one fit on train-normal — test-normal is measurably different from train-normal. **This is the
  reason the threshold is fit on a held-out calib split, not on train:** a train-fit threshold would be far
  too tight and flood holdout with false alarms. The z-score baselines are self-normalising, so their ratio
  is ≈ 1.0 and they are largely immune to this — a point in the baseline's favour.
- **(d) Batch-first — no streaming claim.** The pipeline scores materialised Gold windows offline. It does
  **not** stream and makes **no** real-time / online-inference claim. Latency is reconstructed from the
  batch timeline at 1 Hz, not measured on a live wire. Online serving is deliberately out of scope here.
- **(e) Reconstruction models over-generalise.** An autoencoder trained only on normal can learn to
  reconstruct attack patterns that resemble normal dynamics, suppressing the very error signal it relies
  on. This caps recall (and drives (a)). Whether a different objective or architecture narrows this is open.

Open questions being tracked: a dense-AE ablation as a stronger/simpler comparator; per-process thresholds
for the weak P1/P3 stages; and a window length / stride sweep.

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — package/venv manager:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **make** — used by the shortcuts below (optional; raw `uv` commands shown too)

## Install

```bash
make install    # installs deps + a torch build (auto-detects GPU-enabled environment)
```

Override the torch build with `make install TORCH_EXTRA=cpu` or `make install TORCH_EXTRA=cu126`.

Without `make`:

```bash
uv sync --extra cpu       # or: --extra cu126
```

## Data

Raw datasets are **not committed** (HAI is freely downloadable). Place HAI 21.03 files so they resolve
under `data/raw/hai/hai-21.03/*.csv.gz`, or point `DATA_ROOT` at your location:

```bash
export DATA_ROOT=/path/to/data
```

## Run

```bash
make ingest       # HAI .csv.gz -> data/bronze/hai/*.parquet
make silver       # Bronze -> Silver (drop constants, validate, fingerprint)
make gold         # Silver -> Gold (scale, split, manifest) + verify
make pipeline     # full run: Silver -> Gold -> verify
make baseline     # T3: z-score detector on Gold
make train        # T4: train LSTM-AE (TensorBoard + val-loss checkpoint)
make eval         # T5: evaluate all 3 detectors -> results/eval_metrics.json
make test         # run the test suite
```

The canonical numbers above are produced by `cps-eval --device cpu`; the resolved device is written into
`results/eval_metrics.json` (`eval_device`) so the figures are reproducible and match CI.

Without `make`:

```bash
uv run cps-ingest     # ingest
uv run cps-silver     # silver
uv run cps-gold       # gold
uv run cps-pipeline   # silver -> gold -> verify
uv run cps-baseline   # baseline detector
uv run cps-train      # train LSTM-AE
uv run cps-eval       # evaluation report
uv run pytest -v      # test
```

## Layout

```text
src/cps_anomaly_pipeline/   # installable pipeline package
├── paths.py            # PathConfig — all data/run paths from DATA_ROOT / RUNS_ROOT
├── device.py           # get_device() — GPU / CPU auto-select
├── schema.py           # Silver Pandera schemas + Gold split definitions
├── fingerprint.py      # SHA-1 row-level fingerprinting
├── ingest.py           # HAI .csv.gz -> Bronze
├── silver.py           # Bronze -> Silver
├── gold.py             # Silver -> Gold (scaler, splits, manifest)
├── pipeline.py         # Silver -> Gold -> verify orchestrator
├── baseline.py         # T3: z-score baseline (max|z| + mean z²)
├── windowing.py        # T4: sliding-window Dataset (60 features)
├── model_lstm_ae.py    # T4: LSTM-Autoencoder architecture
├── train.py            # T4: model-agnostic trainer (TensorBoard, val-loss ckpt)
├── scoring.py          # T5: per-window scoring, both detectors, shared stride-1 grid
├── evaluate.py         # T5: raw + point-adjust metrics, per-process recall, latency SLI/SLO
└── eval_report.py      # T5: cps-eval runner -> results/eval_metrics.json + comparison table
tests/                      # pytest suite — synthetic data, no real Gold on runners
├── test_ingest.py          test_silver.py        test_gold.py
├── test_baseline.py        test_windowing.py     test_model_lstm_ae.py
├── test_train.py           test_scoring.py       test_evaluate.py
└── test_eval_report.py
results/
└── eval_metrics.json   # canonical evaluation numbers (eval_device=cpu)
notebooks/
└── eda_hai.ipynb       # EDA driving the Silver/Gold schema
.github/workflows/
└── ci.yml              # lint + full test suite on CPU torch
EDA.md                  # EDA facts driving the Silver schema
Makefile                # install / pipeline / baseline / train / eval / test / lint
pyproject.toml
uv.lock
```