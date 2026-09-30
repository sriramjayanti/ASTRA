"""
IQ Interleaving Layout and Scalar Stream Formatting for ASTRA Engine 7.
Handles conversions between complex NumPy arrays and interleaved scalar streams (IQ vs QI).
"""

from __future__ import annotations

import numpy as np


def format_iq_scalar_stream(
    iq: np.ndarray,
    iq_order: str = "IQ",
) -> np.ndarray:
    """Interleave complex baseband IQ samples into a 1D scalar real array.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        iq_order: Interleaving order ('IQ' or 'QI').

    Returns:
        1D float/real array of length 2*len(iq).
    """
    order = iq_order.upper().strip()
    if order not in ("IQ", "QI"):
        raise ValueError(f"Invalid iq_order: '{iq_order}'. Must be 'IQ' or 'QI'.")

    if len(iq) == 0:
        return np.empty(0, dtype=np.float32)

    n = len(iq)
    scalars = np.empty(2 * n, dtype=np.float32)

    if order == "IQ":
        scalars[0::2] = iq.real
        scalars[1::2] = iq.imag
    else:  # QI
        scalars[0::2] = iq.imag
        scalars[1::2] = iq.real

    return scalars


def deformat_iq_scalar_stream(
    scalars: np.ndarray,
    iq_order: str = "IQ",
) -> np.ndarray:
    """De-interleave a 1D scalar stream back into a complex64 NumPy array.

    Args:
        scalars: 1D real/scalar array of length 2N.
        iq_order: Interleaving order ('IQ' or 'QI').

    Returns:
        1D complex64 array of length N.
    """
    order = iq_order.upper().strip()
    if order not in ("IQ", "QI"):
        raise ValueError(f"Invalid iq_order: '{iq_order}'. Must be 'IQ' or 'QI'.")

    if len(scalars) == 0:
        return np.empty(0, dtype=np.complex64)

    if len(scalars) % 2 != 0:
        raise ValueError(f"Scalar stream length must be even, got {len(scalars)}")

    n = len(scalars) // 2
    if order == "IQ":
        real = scalars[0::2]
        imag = scalars[1::2]
    else:  # QI
        imag = scalars[0::2]
        real = scalars[1::2]

    return (real + 1j * imag).astype(np.complex64)
