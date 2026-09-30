"""
constellation.py
Helper functions and utilities for working with ConstellationDefinition structures.
"""

from typing import Dict, List, Tuple
import numpy as np
from .models import ConstellationDefinition


def get_bit_subsets(
    constellation: ConstellationDefinition
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    For each bit index k in [0, bits_per_symbol - 1], return indices of symbols where:
      - bit_k == 0 (subset S_{k,0})
      - bit_k == 1 (subset S_{k,1})
    """
    subsets = []
    bps = constellation.bits_per_symbol
    labels = constellation.bit_labels  # [M, bps]

    for k in range(bps):
        idx_0 = np.where(labels[:, k] == 0)[0]
        idx_1 = np.where(labels[:, k] == 1)[0]
        subsets.append((idx_0, idx_1))
    return subsets


def rotate_constellation(
    constellation: ConstellationDefinition,
    rotation_deg: float
) -> ConstellationDefinition:
    """Return a rotated copy of the constellation."""
    rot_rad = np.deg2rad(rotation_deg)
    rotator = np.exp(1j * rot_rad)
    new_points = (constellation.complex_points * rotator).astype(np.complex64)

    return ConstellationDefinition(
        name=f"{constellation.name}_rot{int(rotation_deg)}",
        complex_points=new_points,
        bit_labels=constellation.bit_labels.copy(),
        symbol_indices=constellation.symbol_indices.copy(),
        average_energy=constellation.average_energy,
        bits_per_symbol=constellation.bits_per_symbol
    )
