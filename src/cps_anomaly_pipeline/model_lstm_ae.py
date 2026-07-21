"""T4 model — LSTM-Autoencoder for windowed CPS reconstruction.

The autoencoder is trained only on normal windows to minimise reconstruction
error. At scoring time (T5) a window that reconstructs poorly — high MSE — is
flagged anomalous. Because attacks perturb the temporal dynamics of the sensors,
a model that has only ever seen normal dynamics reconstructs attacked windows
worse. This is the mechanism the point-wise z-score baseline (T3) cannot capture:
it sees each row independently, the AE sees the sequence.

Architecture (sequence-to-sequence reconstruction):
  * Encoder: an LSTM over the (window, 60) input; the final hidden state is the
    fixed-size latent summary of the whole window.
  * Bottleneck: that latent vector is repeated across the window length to seed
    the decoder — the model must rebuild the full sequence from one summary,
    which is what forces it to learn the normal dynamics rather than copy.
  * Decoder: an LSTM over the repeated latent, then a linear head mapping hidden
    -> 60 features per timestep.

Kept deliberately small (hidden=64, 1 layer) for fast iteration; device
placement is handled by get_device().
"""

from __future__ import annotations

import torch
from torch import nn

from cps_anomaly_pipeline.windowing import N_FEATURES

DEFAULT_HIDDEN = 64
DEFAULT_LAYERS = 1


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder.

    Input and output are both (batch, window, n_features); the training target is
    the input itself.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden: int = DEFAULT_HIDDEN,
        num_layers: int = DEFAULT_LAYERS,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input window.

        x: (batch, window, n_features) -> same shape.
        """
        batch, window, _ = x.shape

        # Encode: keep the last layer's final hidden state as the latent summary.
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]  # (batch, hidden)

        # Repeat the latent across the window to seed the decoder.
        seed = latent.unsqueeze(1).repeat(1, window, 1)  # (batch, window, hidden)

        decoded, _ = self.decoder(seed)  # (batch, window, hidden)
        return self.output(decoded)  # (batch, window, n_features)


def reconstruction_error(model: LSTMAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Per-window mean squared reconstruction error.

    Returns a (batch,) tensor: the MSE averaged over window and feature dims.
    This is the anomaly score used downstream in T5.
    """
    recon = model(x)
    per_element = (recon - x) ** 2
    return per_element.mean(dim=(1, 2))
