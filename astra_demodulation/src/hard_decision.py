"""
hard_decision.py
Vectorized nearest-neighbor hard decision slicer, adaptive constellation clustering/scaling, and margin calculator.
"""

from typing import Tuple, Optional
import numpy as np
from .models import ConstellationDefinition, HardDecisionResult


def align_and_scale_constellation(
    symbols: np.ndarray,
    constellation: ConstellationDefinition,
    max_iters: int = 5
) -> np.ndarray:
    """
    Decision-directed adaptive gain and phase tracking.
    Refines complex gain g = scale * exp(j * theta) to minimize mean-squared Euclidean distance
    between observed symbol clusters and target theoretical constellation points.
    """
    if len(symbols) < 16:
        return symbols

    aligned_symbols = symbols.copy().astype(np.complex64)
    ref_points = constellation.complex_points

    for _ in range(max_iters):
        # 1. Slice against current estimate
        diffs = aligned_symbols[:, np.newaxis] - ref_points[np.newaxis, :]
        dist_sq = np.real(diffs * np.conj(diffs))
        best_indices = np.argmin(dist_sq, axis=1)
        target_pts = ref_points[best_indices]

        # 2. Compute optimal complex LS scalar g = <y, s> / ||y||^2
        denom = np.sum(np.abs(aligned_symbols) ** 2)
        if denom < 1e-8:
            break
        num = np.sum(target_pts * np.conj(aligned_symbols))
        complex_scale = num / denom

        # Bound gain adjustment to prevent divergence
        abs_scale = np.abs(complex_scale)
        if abs_scale < 0.2 or abs_scale > 5.0:
            break

        # Apply progressive adjustment
        aligned_symbols = (aligned_symbols * complex_scale).astype(np.complex64)
        if abs(abs_scale - 1.0) < 1e-4 and np.abs(np.angle(complex_scale)) < 1e-4:
            break

    return aligned_symbols


def slice_hard_decisions(
    symbols: np.ndarray,
    constellation: ConstellationDefinition,
    chunk_size: int = 65536,
    adaptive_align: bool = True
) -> HardDecisionResult:
    """
    Vectorized nearest-neighbor hard decision demapping with adaptive gain/phase alignment.
    
    Computes Euclidean distance to all M constellation points, finds argmin,
    extracts Gray-coded hard bits, and calculates decision margins.
    """
    if adaptive_align and len(symbols) >= 32:
        proc_symbols = align_and_scale_constellation(symbols, constellation)
    else:
        proc_symbols = symbols

    n_symbols = len(proc_symbols)
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
        chunk_syms = proc_symbols[start:end]  # [Chunk]

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
