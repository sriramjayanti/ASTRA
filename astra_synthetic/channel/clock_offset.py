"""
Sample Clock Frequency Offset (PPM) Impairment Module for ASTRA Engine 6.
Simulates clock mismatch between transmitter and receiver using high-fidelity resampling.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def apply_sample_clock_offset(
    iq: np.ndarray,
    ppm: float,
    sample_rate: float,
) -> tuple[np.ndarray, float]:
    """Apply sample clock frequency offset (drift/mismatch) in parts-per-million.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        ppm: Clock offset in parts-per-million (e.g. +20.0 ppm, -50.0 ppm).
        sample_rate: Nominal sample rate F_s in Hz.

    Returns:
        tuple (resampled_iq, effective_sample_rate)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), sample_rate

    rate_ratio = 1.0 + (ppm * 1e-6)
    effective_sample_rate = float(sample_rate * rate_ratio)

    if abs(ppm) < 1e-4:
        return iq.astype(np.complex64), effective_sample_rate

    # Resample using linear or spline interpolation over non-integer time grid
    n_in = len(iq)
    t_in = np.arange(n_in, dtype=np.float64)
    # Output grid advances with rate_ratio
    n_out = int(np.round(n_in / rate_ratio))
    t_out = np.arange(n_out, dtype=np.float64) * rate_ratio

    # Clip query points to valid range
    valid_mask = t_out < (n_in - 1)
    t_out_valid = t_out[valid_mask]

    # Separable real and imaginary linear interpolation
    real_interp = np.interp(t_out_valid, t_in, iq.real)
    imag_interp = np.interp(t_out_valid, t_in, iq.imag)
    resampled = (real_interp + 1j * imag_interp).astype(np.complex64)

    return resampled, effective_sample_rate
