# cps-anomaly-pipeline

A DE-first anomaly detection pipeline for cyber-physical systems, built on the
[HAI](https://github.com/icsdataset/hai) industrial control-system testbed dataset. The focus is production data
infrastructure — a Medallion-layered data pipeline, schema validation, reproducible runs, and honest evaluation — rather
than a single model in a notebook.

## Status

**T1 complete** — HAI ingestion into the Bronze layer.

| Stage | Description                                               | State |
|-------|-----------------------------------------------------------|-------|
| T1    | Ingest HAI `.csv.gz` → Bronze (Parquet, raw-as-ingested)  | ✅    |
| T2    | Silver / Gold + Pandera validation + SHA-1 fingerprinting | ⬜    |
| T3    | Baseline detector                                         | ⬜    |
| T4    | LSTM-Autoencoder + file-based run registry                | ⬜    |
| T5    | Honest evaluation (per-attack, detection lead time)       | ⬜    |
| T6    | SLO / SLI framing + serving                               | ⬜    |
| T7    | Tests + CI                                                | ⬜    |

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — package/venv manager:
  ```bash
  # macOS / Linux / WSL2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **make** — used by the shortcuts below (optional; raw `uv` commands shown too):
  ```bash
  # Debian / Ubuntu / WSL2
  sudo apt install make
  # macOS (via Xcode command line tools)
  xcode-select --install
  ```

> On Windows, run everything inside **WSL2** — the Makefile needs a POSIX `make`.

## Install

```bash
make install    # detects GPU, installs the matching torch build
```

`make install` auto-selects CUDA (`cu126`) when an NVIDIA GPU is present, otherwise the CPU/MPS build. Override with
`make install TORCH_EXTRA=cpu`.

Without `make`:

```bash
uv sync --extra cpu       # or: --extra cu126 on an NVIDIA machine
```

## Data

Raw datasets are **not committed** (HAI is freely downloadable; SWaT is retained locally and may not be redistributed).
Place HAI 21.03 files so they resolve under `data/raw/hai/hai-21.03/*.csv.gz`, or point `DATA_ROOT` at your location:

```bash
export DATA_ROOT=/path/to/data
```

## Run

```bash
make ingest     # HAI .csv.gz -> data/bronze/hai/*.parquet
make test       # run the test suite
```

Without `make`:

```bash
uv run cps-anomaly-pipeline   # ingest
uv run pytest -v              # test
```

## Layout

```
src/cps_anomaly_pipeline/
  paths.py      PathConfig — all data paths from a single DATA_ROOT
  device.py     get_device() — cuda / mps / cpu auto-select
  ingest.py     HAI -> Bronze
tests/
  test_ingest.py
```