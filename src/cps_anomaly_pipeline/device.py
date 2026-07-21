"""Torch device selection for GPU-enabled and CPU-only environments.

torch is imported lazily so data-only stages do not require torch installed.
"""

from __future__ import annotations


def get_device() -> str:
    """Return the best available torch device string (accelerator if present, else CPU)."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
