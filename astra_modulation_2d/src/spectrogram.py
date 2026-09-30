"""
ASTRA STFT Spectrogram Generator for Complex Baseband IQ.

Features:
1. Differentiable / GPU-accelerated STFT computation
2. Full complex baseband spectrum (two-sided, negative to positive frequencies)
3. Frequency-axis fftshift centering DC (0 Hz) at the midpoint
4. Log-power computation: 10 * log10(|STFT|^2 + eps)
5. Standard per-window standardization or min-max normalization
6. Pure numerical tensor outputs [B, 1, F, T] (No lossy PNG/image conversion)
"""

from __future__ import annotations

import math
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn


class SpectrogramGenerator(nn.Module):
    """
    Transforms 1D complex IQ signals into centered log-power spectrograms.
    """
    def __init__(
        self,
        n_fft: int = 128,
        hop_length: int = 32,
        win_length: Optional[int] = 128,
        window: str = "hann",
        fftshift: bool = True,
        representation: str = "log_power",
        normalization: str = "standard",
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window_type = window.lower()
        self.fftshift_enabled = fftshift
        self.representation = representation.lower()
        self.normalization = normalization.lower()
        self.epsilon = epsilon

        # Register Hann window buffer
        if self.window_type == "hann":
            win_tensor = torch.hann_window(self.win_length)
        elif self.window_type == "hamming":
            win_tensor = torch.hamming_window(self.win_length)
        elif self.window_type == "blackman":
            win_tensor = torch.blackman_window(self.win_length)
        else:
            win_tensor = torch.ones(self.win_length)

        self.register_buffer("window", win_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex IQ tensor [B, N] or real 2-channel tensor [B, 2, N]
            
        Returns:
            spectrogram: Tensor of shape [B, 1, F, T] where F = n_fft
        """
        if x.dim() == 3 and x.size(1) == 2:
            # Convert [B, 2, N] to complex [B, N]
            complex_iq = torch.complex(x[:, 0, :], x[:, 1, :])
        elif x.dim() == 2 and torch.is_complex(x):
            complex_iq = x
        elif x.dim() == 1:
            complex_iq = x.unsqueeze(0)
            if not torch.is_complex(complex_iq):
                complex_iq = complex_iq.to(torch.complex64)
        else:
            raise ValueError(f"Unsupported input shape/type for SpectrogramGenerator: {x.shape}, dtype={x.dtype}")

        # Compute full two-sided complex STFT
        # onesided=False is critical because complex baseband IQ has non-symmetric positive & negative frequencies
        stft_out = torch.stft(
            complex_iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(complex_iq.device),
            center=True,
            normalized=False,
            onesided=False,
            return_complex=True,
        )  # Shape: [B, n_fft, Time_Frames]

        # Shift DC (0 Hz) to the center along frequency axis (dim=1)
        if self.fftshift_enabled:
            stft_out = torch.fft.fftshift(stft_out, dim=1)

        # Compute Power / Log-Power
        power = torch.abs(stft_out) ** 2
        if self.representation == "log_power":
            spec = 10.0 * torch.log10(power + self.epsilon)
        elif self.representation == "magnitude":
            spec = torch.sqrt(power + self.epsilon)
        else:
            spec = power

        # Normalization
        if self.normalization == "standard":
            # Per-window standardization (zero mean, unit variance)
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True) + 1e-6
            spec = (spec - mean) / std
        elif self.normalization == "min_max":
            # Per-window min-max scaling to [0, 1]
            min_val = spec.amin(dim=(-2, -1), keepdim=True)
            max_val = spec.amax(dim=(-2, -1), keepdim=True)
            spec = (spec - min_val) / (max_val - min_val + 1e-6)

        # Return [B, 1, F, T]
        return spec.unsqueeze(1)

    def numpy_to_spectrogram(self, iq_complex: np.ndarray) -> np.ndarray:
        """Helper for numpy evaluation pipelines."""
        tensor_x = torch.from_numpy(iq_complex).unsqueeze(0)
        with torch.no_grad():
            spec_tensor = self.forward(tensor_x)
        return spec_tensor.squeeze(0).squeeze(0).cpu().numpy()
