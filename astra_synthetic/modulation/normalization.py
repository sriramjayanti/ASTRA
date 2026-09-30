"""
Power and energy normalization utilities for ASTRA Engine 5.
Maintains predictable RMS signal power (1.0) across all modulation families.
"""

from __future__ import annotations

import numpy as np


def calculate_average_power(samples: np.ndarray) -> float:
    """Calculate average power (mean squared magnitude) of complex or real samples."""
    if len(samples) == 0:
        return 0.0
    return float(np.mean(np.abs(samples) ** 2))


def normalize_signal_power(
    samples: np.ndarray,
    target_power: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Normalize signal power to target average power (default 1.0).

    Args:
        samples: 1D NumPy array of IQ samples.
        target_power: Desired average power.

    Returns:
        tuple (normalized_samples, scale_factor, measured_pre_power)
    """
    pre_power = calculate_average_power(samples)
    if pre_power <= 1e-12:
        return samples.copy(), 1.0, pre_power

    scale_factor = np.sqrt(target_power / pre_power)
    normalized = (samples * scale_factor).astype(np.complex64)
    return normalized, float(scale_factor), pre_power
