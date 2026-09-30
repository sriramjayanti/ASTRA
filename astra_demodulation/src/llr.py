"""
llr.py
Log-Likelihood Ratio (LLR) computation module supporting Exact (logsumexp) and Max-Log approximations.

ASTRA Sign Convention:
  LLR(b_k) = ln( P(b_k = 0 | y) / P(b_k = 1 | y) )
  LLR > 0 -> Bit 0 favored
  LLR < 0 -> Bit 1 favored
"""

from typing import Tuple, Optional
import numpy as np
from scipy.special import logsumexp
from .models import ConstellationDefinition, SoftDecisionResult
from .constellation import get_bit_subsets


def compute_soft_llrs(
    symbols: np.ndarray,
    constellation: ConstellationDefinition,
    noise_variance: float = 0.05,
    mode: str = "max_log",
    clip_range: Tuple[float, float] = (-30.0, 30.0),
    chunk_size: int = 65536
) -> SoftDecisionResult:
    """
    Compute soft Log-Likelihood Ratios for every bit in the received symbol stream.
    
    Returns:
        SoftDecisionResult with concatenated LLRs [N * bits_per_symbol] and confidences.
    """
    n_symbols = len(symbols)
    ref_points = constellation.complex_points  # [M]
    bps = constellation.bits_per_symbol
    subsets = get_bit_subsets(constellation)   # [(idx_0, idx_1) for each k in range(bps)]

    sigma_sq = max(1e-6, float(noise_variance))
    all_llrs = np.zeros((n_symbols, bps), dtype=np.float32)

    # Process in memory-safe chunks
    for start in range(0, n_symbols, chunk_size):
        end = min(start + chunk_size, n_symbols)
        chunk_syms = symbols[start:end]

        # Squared Euclidean distances to all M reference points: [Chunk, M]
        diffs = chunk_syms[:, np.newaxis] - ref_points[np.newaxis, :]
        dist_sq = np.real(diffs * np.conj(diffs))  # [Chunk, M]

        if mode == "exact":
            # Exponents: - |y - s|^2 / sigma^2
            neg_exponents = -dist_sq / sigma_sq
            for k in range(bps):
                idx_0, idx_1 = subsets[k]
                log_p0 = logsumexp(neg_exponents[:, idx_0], axis=1)
                log_p1 = logsumexp(neg_exponents[:, idx_1], axis=1)
                all_llrs[start:end, k] = (log_p0 - log_p1).astype(np.float32)
        else:
            # Max-Log approximation: (min_dist_1 - min_dist_0) / sigma^2
            for k in range(bps):
                idx_0, idx_1 = subsets[k]
                min_d0 = np.min(dist_sq[:, idx_0], axis=1)
                min_d1 = np.min(dist_sq[:, idx_1], axis=1)
                all_llrs[start:end, k] = ((min_d1 - min_d0) / sigma_sq).astype(np.float32)

    # Apply clipping if enabled
    if clip_range is not None:
        min_c, max_c = clip_range
        all_llrs = np.clip(all_llrs, min_c, max_c)

    flat_llrs = all_llrs.flatten()

    # Normalized bit confidence in [0, 1] via sigmoid-like transformation: 1 - 2/(1 + exp(|LLR|))
    bit_confidences = (1.0 - 2.0 / (1.0 + np.exp(np.minimum(50.0, np.abs(flat_llrs))))).astype(np.float32)
    # Per-symbol confidence is the mean bit confidence of its constituent bits
    symbol_confidences = np.mean(all_llrs.reshape(n_symbols, bps), axis=1).astype(np.float32)

    return SoftDecisionResult(
        llrs=flat_llrs,
        bit_confidences=bit_confidences,
        symbol_confidences=symbol_confidences,
        llr_mode=mode,
        noise_variance=sigma_sq
    )
