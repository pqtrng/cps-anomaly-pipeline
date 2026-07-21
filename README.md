[![CI](https://github.com/pqtrng/cps-anomaly-pipeline/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/pqtrng/cps-anomaly-pipeline/actions/workflows/ci.yml)

# cps-anomaly-pipeline

A DE-first anomaly detection pipeline for cyber-physical systems, built on the
[HAI](https://github.com/icsdataset/hai) industrial control-system testbed dataset. The focus is production data
infrastructure — a Medallion-layered data pipeline, schema validation, reproducible runs, and honest evaluation — rather
than a single model in a notebook.

## Status

**T4 complete** — LSTM-Autoencoder trained on normal-only windows, with a z-score baseline for comparison, TensorBoard tracking, and val-loss checkpoint selection.

| Stage | Description                                               | State |
| ----- | --------------------------------------------------------- | ----- |
| T1    | Ingest HAI `.csv.gz` → Bronze (Parquet, raw-as-ingested)  | ✅     |
| T2    | Silver / Gold + Pandera validation + SHA-1 fingerprinting | ✅     |
| T3    | Baseline detector (z-score, both scoring methods)         | ✅     |
| T4    | LSTM-Autoencoder (windowing, TensorBoard, val-loss ckpt)  | ✅     |
| T5    | Honest evaluation (per-attack, detection lead time)       | ⬜     |
| T6    | SLO / SLI framing + serving                               | ⬜     |
| T7    | Tests + CI                                                | ✅     |

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — package/venv manager:
  
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **make** — used by the shortcuts below (optional; raw `uv` commands shown too)

## Install

```bash
make install    # installs deps + a torch build (see Makefile TORCH_EXTRA)
```

Override the torch build with `make install TORCH_EXTRA=cpu` or
`make install TORCH_EXTRA=cu126`.

Without `make`:

```bash
uv sync --extra cpu       # or: --extra cu126
```

## Data

Raw datasets are **not committed** (HAI is freely downloadable; SWaT is retained locally and may not be redistributed).
Place HAI 21.03 files so they resolve under `data/raw/hai/hai-21.03/*.csv.gz`, or point `DATA_ROOT` at your location:

```bash
export DATA_ROOT=/path/to/data
```

## Run

```bash
make ingest       # HAI .csv.gz -> data/bronze/hai/*.parquet
make silver       # Bronze -> Silver (drop constants, validate, fingerprint)
make gold         # Silver -> Gold (scale, split, manifest) + verify
make pipeline     # Full run: Silver -> Gold -> verify
make test         # run the test suite
```

Without `make`:

```bash
uv run cps-ingest     # ingest
uv run cps-silver     # silver
uv run cps-gold       # gold
uv run cps-pipeline   # silver -> gold -> verify
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
└── train.py            # T4: model-agnostic trainer (TensorBoard, val-loss ckpt)
tests/                      # pytest suite — synthetic data, no real Gold on runners
├── test_ingest.py
├── test_silver.py
├── test_gold.py
├── test_baseline.py
├── test_windowing.py
├── test_model_lstm_ae.py
└── test_train.py
notebooks/
└── eda_hai.ipynb       # EDA driving the Silver/Gold schema
.github/workflows/
└── ci.yml              # lint + full test suite on CPU torch
EDA.md                  # EDA facts driving the Silver schema
Makefile                # install / pipeline / baseline / train / test / lint
pyproject.toml
uv.lock
```