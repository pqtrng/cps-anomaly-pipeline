# Pick torch wheels: cu126 on a GPU-enabled environment, else cpu.
# Override anytime, e.g.:  make install TORCH_EXTRA=cpu
TORCH_EXTRA ?= $(shell command -v nvidia-smi >/dev/null 2>&1 && echo cu126 || echo cpu)
MODEL ?= lstm_ae

.DEFAULT_GOAL := help
.PHONY: help install ingest silver gold pipeline baseline train save-metrics test lint fix

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install deps + matching torch (auto-detects GPU) + train tooling
	uv sync --extra $(TORCH_EXTRA) --extra train

ingest:  ## Ingest HAI .csv.gz -> Bronze Parquet
	uv run cps-ingest

silver:  ## Bronze -> Silver (drop constants, validate, fingerprint)
	uv run cps-silver

gold:  ## Silver -> Gold (scale, split, manifest) + verify
	uv run cps-gold

pipeline:  ## Full run: Silver -> Gold -> verify
	uv run cps-pipeline

baseline:  ## T3 z-score baseline detector (both scoring methods) on Gold
	uv run cps-baseline

train:  ## T4 train LSTM-AE (TensorBoard + val_loss checkpoint)
	uv run cps-train

save-metrics:  ## Copy run metrics.json into Git-tracked results/ (MODEL=lstm_ae)
	@mkdir -p results
	@cp runs/$(MODEL)/metrics.json results/$(MODEL)_metrics.json
	@echo "Saved results/$(MODEL)_metrics.json — commit this to version the numbers."

test:  ## Run the test suite
	uv run pytest -v

lint:  ## Run ruff (auto-fix) + yamllint
	uv run ruff check --fix .
	uv run yamllint .

fix:  ## Alias for lint (auto-fixes what ruff can)
	uv run ruff check --fix .