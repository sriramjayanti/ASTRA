"""
sync_candidates.py
Blind sync word discovery and candidate sync scoring without prior pattern knowledge.
"""

from typing import List, Optional
import numpy as np

from .models import SyncPatternResult, PeriodicityCandidate
from .repeated_patterns import analyze_repeated_prefixes, find_repeated_patterns


def discover_candidate_sync(
    bits: np.ndarray,
    candidate_lengths: List[int] = None,
    candidate_frame_lengths: List[PeriodicityCandidate] = None,
    soft_info: Optional[np.ndarray] = None
) -> List[SyncPatternResult]:
    """
    Discover candidate sync words blindly by analyzing repeated prefixes across candidate frame lengths
    and frequent recurring bit sequences.

    Args:
        bits: 1D uint8 array of binary bits.
        candidate_lengths: Sync pattern lengths to probe (e.g. [16, 24, 32, 64]).
        candidate_frame_lengths: Top frame length hypotheses.
        soft_info: Optional soft reliability array.

    Returns:
        List of SyncPatternResult representing blind sync hypotheses.
    """
    if candidate_lengths is None:
        candidate_lengths = [16, 24, 32, 64]

    discovered: List[SyncPatternResult] = []
    seen_patterns = set()

    # 1. Probe candidate frame lengths for repeated prefixes
    if candidate_frame_lengths:
        for fl_cand in candidate_frame_lengths[:3]:
            fl = fl_cand.period_bits
            repeated_prefixes = analyze_repeated_prefixes(
                bits,
                frame_length=fl,
                prefix_lengths=candidate_lengths
            )

            for rp in repeated_prefixes:
                if rp.pattern_bits in seen_patterns:
                    continue
                seen_patterns.add(rp.pattern_bits)

                pat_bits = np.array([int(b) for b in rp.pattern_bits], dtype=np.uint8)
                
                # Periodicity consistency check
                spacing_match = False
                if rp.mean_spacing > 0:
                    spacing_match = abs(rp.mean_spacing - fl) / max(1.0, fl) < 0.05

                periodicity_score = 0.9 if (spacing_match and rp.spacing_variance < 1.0) else 0.5
                confidence = float(np.clip(
                    0.5 * (rp.occurrence_count / max(1.0, fl_cand.estimated_frame_count or 10)) + 0.5 * periodicity_score,
                    0.0, 1.0
                ))

                discovered.append(SyncPatternResult(
                    pattern_id=f"blind_prefix_L{rp.length}_FL{fl}",
                    bit_pattern=pat_bits.tolist(),
                    length=rp.length,
                    positions=rp.positions,
                    match_scores=[1.0] * len(rp.positions),
                    mean_spacing=rp.mean_spacing,
                    spacing_variance=rp.spacing_variance,
                    match_count=rp.occurrence_count,
                    periodicity_score=periodicity_score,
                    confidence=confidence
                ))

    # 2. General repeated patterns with regular spacing
    general_repeated = find_repeated_patterns(
        bits,
        pattern_lengths=[l for l in candidate_lengths if l in [16, 24, 32]],
        min_occurrences=3
    )

    for gr in general_repeated:
        if gr.pattern_bits in seen_patterns:
            continue

        # If variance is low, it occurs at periodic intervals!
        if gr.occurrence_count >= 3 and gr.spacing_variance < 50.0 and gr.mean_spacing > 32:
            seen_patterns.add(gr.pattern_bits)
            pat_bits = np.array([int(b) for b in gr.pattern_bits], dtype=np.uint8)

            std_sp = np.sqrt(gr.spacing_variance)
            cv = std_sp / gr.mean_spacing
            periodicity_score = float(np.clip(1.0 - cv, 0.0, 1.0))
            confidence = float(np.clip(0.4 * min(1.0, gr.occurrence_count / 10.0) + 0.6 * periodicity_score, 0.0, 1.0))

            discovered.append(SyncPatternResult(
                pattern_id=f"blind_recurring_L{gr.length}_period{int(gr.mean_spacing)}",
                bit_pattern=pat_bits.tolist(),
                length=gr.length,
                positions=gr.positions,
                match_scores=[1.0] * len(gr.positions),
                mean_spacing=gr.mean_spacing,
                spacing_variance=gr.spacing_variance,
                match_count=gr.occurrence_count,
                periodicity_score=periodicity_score,
                confidence=confidence
            ))

    # Sort discovered by confidence descending
    discovered.sort(key=lambda s: s.confidence, reverse=True)
    return discovered
