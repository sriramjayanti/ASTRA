"""
Centralized IQ Preprocessing for ASTRA 2D Spectrogram Classifier.

Ensures strict mathematical alignment with 1D raw-IQ preprocessing:
1. Finite validation (NaN/Inf detection and handling)
2. Zero-mean DC offset removal: x = x - mean(x)
3. RMS amplitude normalization: x = x / (sqrt(mean(|x|^2)) + eps)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import torch


class IQPreprocessor:
    """
    Centralized IQ Preprocessor for complex baseband signals.
    Supports NumPy arrays [N] or PyTorch tensors [..., N].
    """
    def __init__(
        self,
        remove_dc: bool = True,
        rms_normalization: bool = True,
        strict_finite: bool = True,
        epsilon: float = 1e-8,
    ):
        self.remove_dc = remove_dc
        self.rms_normalization = rms_normalization
        self.strict_finite = strict_finite
        self.epsilon = epsilon

    def process_numpy(self, iq: np.ndarray) -> np.ndarray:
        """
        Processes 1D complex NumPy array [N].
        """
        if not np.iscomplexobj(iq):
            iq = iq.astype(np.complex64)

        if not np.all(np.isfinite(iq)):
            if self.strict_finite:
                iq = np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError("Signal contains NaN or Inf values.")

        if self.remove_dc:
            iq = iq - np.mean(iq)

        if self.rms_normalization:
            rms = np.sqrt(np.mean(np.abs(iq) ** 2))
            iq = iq / (rms + self.epsilon)

        return iq.astype(np.complex64)

    def process_torch(self, iq: torch.Tensor) -> torch.Tensor:
        """
        Processes 1D complex PyTorch tensor [..., N].
        """
        if not torch.is_complex(iq):
            iq = iq.to(torch.complex64)

        if not torch.all(torch.isfinite(iq)):
            if self.strict_finite:
                iq = torch.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError("Signal contains NaN or Inf values.")

        if self.remove_dc:
            iq = iq - torch.mean(iq, dim=-1, keepdim=True)

        if self.rms_normalization:
            rms = torch.sqrt(torch.mean(torch.abs(iq) ** 2, dim=-1, keepdim=True))
            iq = iq / (rms + self.epsilon)

        return iq

    def __call__(self, iq: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(iq, torch.Tensor):
            return self.process_torch(iq)
        return self.process_numpy(iq)
