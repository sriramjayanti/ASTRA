"""
Carrier Phase Offset Impairment Module for ASTRA Engine 6.
Applies constant carrier phase rotation exp(j * phi) to complex baseband IQ.
"""

from __future__ import annotations

import numpy as np


def apply_phase_offset(
    iq: np.ndarray,
    phase_offset_rad: float,
) -> np.ndarray:
    """Apply constant carrier phase offset rotation.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        phase_offset_rad: Phase rotation in radians [-pi, pi].

    Returns:
        1D complex64 array with phase rotation applied.
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64)

    if abs(phase_offset_rad) < 1e-12:
        return iq.astype(np.complex64)

    rotation = np.exp(1j * phase_offset_rad)
    return (iq * rotation).astype(np.complex64)
