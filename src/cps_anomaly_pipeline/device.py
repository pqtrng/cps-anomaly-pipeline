"""Device selection compatible with CUDA and CPU.

torch is imported lazily so data-only stages do not require torch installed.
"""

from __future__ import annotations


def get_device() -> str:
    """Return the best available torch device string: cuda > mps > cpu.

    Auto-selection avoids hardcoding a device so the same code runs on the GPU
    machine and on a laptop.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
