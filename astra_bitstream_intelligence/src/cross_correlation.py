"""
cross_correlation.py
Cross-correlation and pattern matching with known and candidate sync words.
Supports exact matching, Hamming-tolerant matching, and soft-weighted correlation.
"""

from typing import List, Union, Optional
import numpy as np

from .models import SyncPatternResult


def hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert hex string (e.g. 'EB90', '0x1ACFFC1D') to uint8 bit array."""
    clean_hex = hex_str.strip().lower()
    if clean_hex.startswith("0x"):
        clean_hex = clean_hex[2:]

    bits = []
    for char in clean_hex:
        val = int(char, 16)
        for shift in (3, 2, 1, 0):
            bits.append((val >> shift) & 1)

    return np.array(bits, dtype=np.uint8)


def search_known_sync(
    bits: np.ndarray,
    pattern: Union[str, np.ndarray, List[int]],
    pattern_id: str = "known_sync",
    max_hamming_fraction: float = 0.0,
    soft_info: Optional[np.ndarray] = None
) -> SyncPatternResult:
    """
    Search for a sync word across the entire bitstream.

    Args:
        bits: 1D uint8 array of bitstream bits.
        pattern: Hex string or uint8 array of sync word bits.
        pattern_id: Identifier label for the pattern.
        max_hamming_fraction: Max allowed bit mismatch fraction (0.0 = exact matches only).
        soft_info: Optional 1D soft reliability / LLR array.

    Returns:
        SyncPatternResult with occurrence positions, spacing statistics, and confidence score.
    """
    if isinstance(pattern, str):
        pat_bits = hex_to_bits(pattern)
    elif isinstance(pattern, list):
        pat_bits = np.array(pattern, dtype=np.uint8)
    else:
        pat_bits = pattern.astype(np.uint8)

    pat_len = len(pat_bits)
    n_bits = len(bits)

    if pat_len == 0 or n_bits < pat_len:
        return SyncPatternResult(
            pattern_id=pattern_id,
            bit_pattern=pat_bits.tolist(),
            length=pat_len,
            positions=[],
            match_scores=[],
            match_count=0
        )

    max_errors = int(np.floor(pat_len * max_hamming_fraction)) if max_hamming_fraction > 0 else 0
    positions: List[int] = []
    match_scores: List[float] = []

    # Slide pattern across bitstream
    for pos in range(n_bits - pat_len + 1):
        window = bits[pos:pos + pat_len]
        hamming_dist = int(np.sum(window != pat_bits))

        if hamming_dist <= max_errors:
            # Score: 1.0 for exact, decaying with Hamming distance
            hard_score = float(1.0 - (hamming_dist / pat_len))

            if soft_info is not None:
                soft_window = soft_info[pos:pos + pat_len]
                bipolar_pat = 2.0 * pat_bits.astype(np.float32) - 1.0
                soft_corr = float(np.mean(bipolar_pat * np.sign(soft_window)))
                combined_score = 0.5 * hard_score + 0.5 * float(np.clip(soft_corr, 0.0, 1.0))
            else:
                combined_score = hard_score

            positions.append(pos)
            match_scores.append(round(combined_score, 4))

    # If both exact (1.0) and degraded matches were found, and exact matches form a periodic train,
    # filter out isolated random degraded matches
    if len(positions) > 0 and max_errors > 0:
        exact_pos = [p for p, s in zip(positions, match_scores) if s == 1.0]
        if len(exact_pos) >= 2:
            exact_sp = np.diff(exact_pos)
            if np.var(exact_sp) < 5.0: # Highly periodic exact matches
                exact_scores = [1.0] * len(exact_pos)
                positions = exact_pos
                match_scores = exact_scores

    match_count = len(positions)
    mean_spacing = 0.0
    spacing_var = 0.0
    periodicity_score = 0.0

    if match_count >= 2:
        spacings = np.diff(positions)
        mean_spacing = float(np.mean(spacings))
        spacing_var = float(np.var(spacings))

        if mean_spacing > 0:
            std_spacing = np.sqrt(spacing_var)
            cv = std_spacing / mean_spacing
            periodicity_score = float(np.clip(1.0 - cv, 0.0, 1.0))
    elif match_count == 1:
        periodicity_score = 0.3

    confidence = 0.0
    if match_count > 0:
        mean_match_quality = float(np.mean(match_scores))
        confidence = float(np.clip(0.6 * mean_match_quality + 0.4 * periodicity_score, 0.0, 1.0))

    return SyncPatternResult(
        pattern_id=pattern_id,
        bit_pattern=pat_bits.tolist(),
        length=pat_len,
        positions=positions,
        match_scores=match_scores,
        mean_spacing=mean_spacing,
        spacing_variance=spacing_var,
        match_count=match_count,
        periodicity_score=periodicity_score,
        confidence=confidence
    )
