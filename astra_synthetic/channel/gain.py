"""
Signal Gain / Attenuation Impairment Module for ASTRA Engine 6.
Applies linear or dB amplitude scaling to complex baseband IQ.
"""

from __future__ import annotations

import numpy as np


def apply_gain(
    iq: np.ndarray,
    gain_db: float | None = None,
    gain_linear: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Apply amplitude gain or attenuation to baseband IQ.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        gain_db: Amplitude scaling in dB (e.g. +6.0 dB, -3.0 dB).
        gain_linear: Linear amplitude multiplier.

    Returns:
        tuple (scaled_iq, resolved_gain_db, resolved_gain_linear)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), 0.0, 1.0

    if gain_linear is not None:
        resolved_linear = float(gain_linear)
        resolved_db = float(20.0 * np.log10(max(1e-12, resolved_linear)))
    elif gain_db is not None:
        resolved_db = float(gain_db)
        resolved_linear = float(10.0 ** (resolved_db / 20.0))
    else:
        resolved_db = 0.0
        resolved_linear = 1.0

    scaled_iq = (iq * resolved_linear).astype(np.complex64)
    return scaled_iq, resolved_db, resolved_linear
