"""
bit_balance.py
Global bit balance and zero/one distribution analysis for ASTRA Stage 12.
"""

import numpy as np
from .models import BitBalance


def calculate_bit_balance(bits: np.ndarray) -> BitBalance:
    """
    Calculate zero fraction, one fraction, and balance imbalance.

    Args:
        bits: 1D uint8 array of binary bits.

    Returns:
        BitBalance dataclass instance.
    """
    if bits is None or len(bits) == 0:
        return BitBalance(zero_fraction=0.5, one_fraction=0.5, imbalance=0.0)

    p1 = float(np.mean(bits))
    p0 = float(1.0 - p1)
    imbalance = float(abs(p1 - 0.5))

    return BitBalance(
        zero_fraction=p0,
        one_fraction=p1,
        imbalance=imbalance
    )
