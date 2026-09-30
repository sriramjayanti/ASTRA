"""
Carrier Frequency Offset (CFO) Impairment Module for ASTRA Engine 6.
Applies continuous carrier frequency rotation exp(j * 2 * pi * CFO * n / Fs).
"""

from __future__ import annotations

import numpy as np


def apply_cfo(
    iq: np.ndarray,
    cfo_hz: float,
    sample_rate: float,
    initial_phase: float = 0.0,
) -> np.ndarray:
    """Apply Carrier Frequency Offset to complex baseband IQ.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        cfo_hz: Frequency offset in Hz.
        sample_rate: Sampling rate F_s in Hz.
        initial_phase: Initial starting phase in radians.

    Returns:
        1D complex64 NumPy array with CFO rotation applied.
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64)

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be > 0, got {sample_rate}")

    if abs(cfo_hz) < 1e-12:
        return iq.astype(np.complex64)

    normalized_cfo = float(cfo_hz / sample_rate)
    n = np.arange(len(iq), dtype=np.float64)
    # Phase trajectory: phi[n] = 2 * pi * (cfo_hz / sample_rate) * n + initial_phase
    phase = 2.0 * np.pi * normalized_cfo * n + initial_phase
    rotation = np.exp(1j * phase)

    cfo_iq = (iq * rotation).astype(np.complex64)
    return cfo_iq
