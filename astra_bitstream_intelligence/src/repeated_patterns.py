"""
repeated_patterns.py
Efficient repeated bit-pattern discovery using rolling hash and substring frequency analysis.
"""

from typing import List, Dict
import numpy as np
from collections import defaultdict

from .models import RepeatedPattern


def find_repeated_patterns(
    bits: np.ndarray,
    pattern_lengths: List[int] = None,
    min_occurrences: int = 3,
    max_patterns_per_len: int = 5
) -> List[RepeatedPattern]:
    """
    Discover recurring bit patterns across the bitstream.

    Args:
        bits: 1D uint8 array of binary bits.
        pattern_lengths: List of pattern lengths in bits (e.g. [8, 16, 24, 32, 64]).
        min_occurrences: Minimum times a pattern must appear.
        max_patterns_per_len: Cap on returned patterns per length.

    Returns:
        List of RepeatedPattern instances sorted by occurrence count and length.
    """
    if pattern_lengths is None:
        pattern_lengths = [8, 16, 24, 32, 64]

    n = len(bits)
    all_repeated: List[RepeatedPattern] = []

    for L in pattern_lengths:
        if n < L * min_occurrences:
            continue

        # Use bytes-like representation for fast dict hashing
        # Form sliding substrings
        pattern_positions = defaultdict(list)
        
        # Step through bitstream. For large streams, use step=1 or step=8
        step = 1 if n <= 16384 else (8 if L >= 32 else 1)
        for i in range(0, n - L + 1, step):
            sub_key = bits[i:i + L].tobytes()
            pattern_positions[sub_key].append(i)

        # Filter by min_occurrences
        candidates = [
            (k, pos_list) for k, pos_list in pattern_positions.items()
            if len(pos_list) >= min_occurrences
        ]

        # Sort by occurrence count descending
        candidates.sort(key=lambda item: len(item[1]), reverse=True)

        for raw_bytes, positions in candidates[:max_patterns_per_len]:
            pat_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            pat_str = "".join(str(b) for b in pat_arr)

            spacings = np.diff(positions)
            mean_sp = float(np.mean(spacings)) if len(spacings) > 0 else 0.0
            var_sp = float(np.var(spacings)) if len(spacings) > 0 else 0.0

            all_repeated.append(RepeatedPattern(
                pattern_bits=pat_str,
                length=L,
                occurrence_count=len(positions),
                positions=positions,
                mean_spacing=mean_sp,
                spacing_variance=var_sp
            ))

    # Sort overall by (occurrence_count * length) descending
    all_repeated.sort(key=lambda p: (p.occurrence_count * p.length), reverse=True)
    return all_repeated


def analyze_repeated_prefixes(
    bits: np.ndarray,
    frame_length: int,
    prefix_lengths: List[int] = None,
    max_frames: int = 64
) -> List[RepeatedPattern]:
    """
    Analyze candidate frame boundaries to discover stable repeating frame headers/prefixes.

    Args:
        bits: 1D uint8 array of binary bits.
        frame_length: Candidate frame length in bits.
        prefix_lengths: List of prefix lengths to evaluate (e.g. [16, 24, 32, 64]).
        max_frames: Max frames to inspect.

    Returns:
        List of candidate repeating prefix patterns.
    """
    if prefix_lengths is None:
        prefix_lengths = [16, 24, 32, 64]

    n = len(bits)
    num_frames = min(n // frame_length, max_frames)
    if num_frames < 3:
        return []

    prefix_candidates: List[RepeatedPattern] = []

    for p_len in prefix_lengths:
        if p_len >= frame_length:
            continue

        prefix_counts = defaultdict(list)
        for frame_idx in range(num_frames):
            start = frame_idx * frame_length
            prefix_bytes = bits[start:start + p_len].tobytes()
            prefix_counts[prefix_bytes].append(start)

        # Find most frequent prefix
        if not prefix_counts:
            continue

        best_prefix_bytes, positions = max(prefix_counts.items(), key=lambda item: len(item[1]))
        count = len(positions)

        # If prefix repeats in at least 50% of candidate frames
        if count >= max(2, int(0.5 * num_frames)):
            pat_arr = np.frombuffer(best_prefix_bytes, dtype=np.uint8)
            pat_str = "".join(str(b) for b in pat_arr)
            spacings = np.diff(positions)
            mean_sp = float(np.mean(spacings)) if len(spacings) > 0 else float(frame_length)
            var_sp = float(np.var(spacings)) if len(spacings) > 0 else 0.0

            prefix_candidates.append(RepeatedPattern(
                pattern_bits=pat_str,
                length=p_len,
                occurrence_count=count,
                positions=positions,
                mean_spacing=mean_sp,
                spacing_variance=var_sp
            ))

    return prefix_candidates
