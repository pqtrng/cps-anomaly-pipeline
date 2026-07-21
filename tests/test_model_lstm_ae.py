"""Tests for T4 LSTM-Autoencoder architecture.

Shape and behaviour checks only — no training, so these run in under a second on
CPU:
  1. forward output shape equals input shape;
  2. reconstruction_error returns one score per window;
  3. an untrained model still produces finite, non-negative errors;
  4. the model reconstructs 60 features (matches windowing.N_FEATURES).
"""

from __future__ import annotations

import torch

from cps_anomaly_pipeline.model_lstm_ae import (
    LSTMAutoencoder,
    reconstruction_error,
)
from cps_anomaly_pipeline.windowing import N_FEATURES


def test_forward_shape_matches_input():
    model = LSTMAutoencoder(n_features=N_FEATURES, hidden=16)
    x = torch.randn(8, 60, N_FEATURES)
    out = model(x)
    assert tuple(out.shape) == (8, 60, N_FEATURES)


def test_reconstruction_error_shape():
    model = LSTMAutoencoder(n_features=N_FEATURES, hidden=16)
    x = torch.randn(5, 60, N_FEATURES)
    err = reconstruction_error(model, x)
    assert tuple(err.shape) == (5,)
    assert torch.isfinite(err).all()
    assert (err >= 0).all()


def test_model_features_match_windowing():
    model = LSTMAutoencoder()
    assert model.n_features == N_FEATURES


def test_error_nonzero_on_random_input():
    """An untrained model won't reconstruct random input perfectly -> error > 0."""
    torch.manual_seed(0)
    model = LSTMAutoencoder(n_features=N_FEATURES, hidden=16)
    x = torch.randn(4, 60, N_FEATURES)
    err = reconstruction_error(model, x)
    assert err.mean() > 0


def test_variable_batch_and_window():
    """Model must handle arbitrary batch and window sizes (no hardcoded dims)."""
    model = LSTMAutoencoder(n_features=N_FEATURES, hidden=8)
    for batch, window in [(1, 30), (3, 60), (2, 120)]:
        x = torch.randn(batch, window, N_FEATURES)
        out = model(x)
        assert tuple(out.shape) == (batch, window, N_FEATURES)
