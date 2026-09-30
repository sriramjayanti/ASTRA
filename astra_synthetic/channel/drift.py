"""
Carrier Frequency Drift Impairment Module for ASTRA Engine 6.
Simulates time-varying carrier frequency changes by integrating instantaneous quadratic phase.
"""

from __future__ import annotations

import numpy as np


def apply_frequency_drift(
    iq: np.ndarray,
    drift_hz_per_sec: float,
    sample_rate: float,
    initial_cfo_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply linear frequency drift over time.

    Instantaneous frequency: f(t) = f_0 + alpha * t
    Instantaneous phase: phi(t) = 2 * pi * (f_0 * t + 0.5 * alpha * t^2)

    Args:
        iq: 1D complex NumPy array of IQ samples.
        drift_hz_per_sec: Linear drift rate alpha in Hz/second.
        sample_rate: Sampling frequency F_s in Hz.
        initial_cfo_hz: Initial frequency offset f_0 in Hz.

    Returns:
        tuple (drift_iq, phase_trajectory)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), np.empty(0, dtype=np.float64)

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be > 0, got {sample_rate}")

    if abs(drift_hz_per_sec) < 1e-12 and abs(initial_cfo_hz) < 1e-12:
        return iq.astype(np.complex64), np.zeros(len(iq), dtype=np.float64)

    n = np.arange(len(iq), dtype=np.float64)
    t = n / sample_rate  # time in seconds

    # phi(t) = 2 * pi * (initial_cfo_hz * t + 0.5 * drift_hz_per_sec * t^2)
    phase = 2.0 * np.pi * (initial_cfo_hz * t + 0.5 * drift_hz_per_sec * (t ** 2))
    rotation = np.exp(1j * phase)

    drift_iq = (iq * rotation).astype(np.complex64)
    return drift_iq, phase
