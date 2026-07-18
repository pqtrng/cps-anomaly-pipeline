# Pick the right PyTorch build for THIS machine automatically:
#   - NVIDIA GPU present (nvidia-smi on PATH) -> CUDA wheels  (cu126)
#   - otherwise (e.g. macOS / no GPU)         -> CPU wheels   (cpu)
# Override anytime, e.g.:  make test TORCH_EXTRA=cpu
TORCH_EXTRA ?= $(shell command -v nvidia-smi >/dev/null 2>&1 && echo cu126 || echo cpu)

.DEFAULT_GOAL := help
.PHONY: help install ingest test lint

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install deps + matching torch (auto-detects GPU)
	uv sync --extra $(TORCH_EXTRA)

ingest:  ## Ingest HAI .csv.gz -> Bronze Parquet
	uv run cps-anomaly-pipeline

test:  ## Run the test suite
	uv run pytest -v

lint:  ## Run ruff + yamllint
	uv run ruff check .
	uv run yamllint .