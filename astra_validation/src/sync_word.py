"""
sync_word.py
Sync word detection and spacing periodicity analysis for ASTRA Stage 10.
Supports exact pattern matching and Hamming distance-tolerant search.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from .models import SyncWordResult, EvidenceCheckState


def hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert a hex string (e.g. 'EB90') into a uint8 1D bit array."""
    cleaned = hex_str.strip().replace("0x", "").replace(" ", "")
    byte_vals = bytes.fromhex(cleaned)
    bits = np.unpackbits(np.frombuffer(byte_vals, dtype=np.uint8))
    return bits


def score_sync_periodicity(positions: List[int], total_length: int) -> float:
    """
    Score the regular spacing of sync-word occurrences.
    If positions occur at fixed regular intervals (e.g. 0, 512, 1024), periodicity is 1.0.
    """
    if len(positions) < 2:
        return 0.0

    diffs = np.diff(positions)
    if len(diffs) == 0:
        return 0.0

    # Most common interval
    median_diff = float(np.median(diffs))
    if median_diff <= 0:
        return 0.0

    # Variation around the median
    rel_devs = np.abs(diffs - median_diff) / median_diff
    consistency = float(np.mean(rel_devs < 0.05))

    # Bonus for more repetitions
    repetition_factor = min(1.0, len(positions) / 4.0)
    return float(consistency * repetition_factor)


def search_sync_word(
    bitstream: Union[np.ndarray, List[int]],
    sync_pattern: Union[str, np.ndarray, List[int]],
    pattern_id: str = "custom_sync",
    max_hamming_fraction: float = 0.125,
) -> SyncWordResult:
    """
    Search for occurrences of sync_pattern in bitstream with up to max_hamming_fraction bit errors.
    """
    bits = np.asarray(bitstream, dtype=np.uint8).ravel()
    if isinstance(sync_pattern, str):
        sync_bits = hex_to_bits(sync_pattern)
    else:
        sync_bits = np.asarray(sync_pattern, dtype=np.uint8).ravel()

    m = len(sync_bits)
    n = len(bits)

    if n < m:
        return SyncWordResult(
            pattern_id=pattern_id,
            check_state=EvidenceCheckState.NOT_TESTED,
        )

    max_dist = int(np.floor(m * max_hamming_fraction))
    positions = []
    distances = []

    # Slide window across bitstream
    for i in range(n - m + 1):
        window = bits[i:i + m]
        dist = int(np.sum(window != sync_bits))
        if dist <= max_dist:
            positions.append(i)
            distances.append(dist)

    match_count = len(positions)
    mean_dist = float(np.mean(distances)) if distances else 0.0
    periodicity = score_sync_periodicity(positions, n)

    check_state = EvidenceCheckState.NOT_TESTED
    if match_count >= 2 and periodicity >= 0.40:
        check_state = EvidenceCheckState.PASS
    elif match_count >= 1 and mean_dist <= max_dist:
        check_state = EvidenceCheckState.PASS
    elif match_count == 0:
        check_state = EvidenceCheckState.NOT_TESTED

    return SyncWordResult(
        pattern_id=pattern_id,
        match_count=match_count,
        positions=positions,
        mean_hamming_distance=mean_dist,
        periodicity_score=periodicity,
        check_state=check_state,
    )
