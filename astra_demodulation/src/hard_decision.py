"""
hard_decision.py
Vectorized nearest-neighbor hard decision slicer and margin calculator.
"""

from typing import Tuple, Optional
import numpy as np
from .models import ConstellationDefinition, HardDecisionResult


def slice_hard_decisions(
    symbols: np.ndarray,
    constellation: ConstellationDefinition,
    chunk_size: int = 65536
) -> HardDecisionResult:
    """
    Vectorized nearest-neighbor hard decision demapping.
    
    Computes Euclidean distance to all M constellation points, finds argmin,
    extracts Gray-coded hard bits, and calculates decision margins.
    """
    n_symbols = len(symbols)
    ref_points = constellation.complex_points  # [M]
    bit_labels = constellation.bit_labels      # [M, bps]
    m_order = len(ref_points)
    bps = constellation.bits_per_symbol

    all_indices = np.zeros(n_symbols, dtype=np.int32)
    all_dists = np.zeros(n_symbols, dtype=np.float32)
    all_margins = np.zeros(n_symbols, dtype=np.float32)

    # Process in memory-safe chunks
    for start in range(0, n_symbols, chunk_size):
        end = min(start + chunk_size, n_symbols)
        chunk_syms = symbols[start:end]  # [Chunk]

        # Distance matrix: |y[n] - s_m|^2 -> [Chunk, M]
        diffs = chunk_syms[:, np.newaxis] - ref_points[np.newaxis, :]
        dist_sq = np.real(diffs * np.conj(diffs))

        # Best index and minimum distance
        sorted_indices = np.argsort(dist_sq, axis=1)
        best_idx = sorted_indices[:, 0]
        second_idx = sorted_indices[:, 1] if m_order > 1 else best_idx

        d1 = np.sqrt(dist_sq[np.arange(len(chunk_syms)), best_idx])
        d2 = np.sqrt(dist_sq[np.arange(len(chunk_syms)), second_idx]) if m_order > 1 else d1

        all_indices[start:end] = best_idx
        all_dists[start:end] = d1
        all_margins[start:end] = d2 - d1

    # Map symbol indices to concatenated hard bitstream [N * bps]
    hard_bits = bit_labels[all_indices].flatten().astype(np.uint8)
    nearest_points = ref_points[all_indices]

    return HardDecisionResult(
        symbol_indices=all_indices,
        hard_bits=hard_bits,
        nearest_points=nearest_points,
        decision_distances=all_dists,
        decision_margins=all_margins
    )
