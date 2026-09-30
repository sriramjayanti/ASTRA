"""
frame_length.py
Multi-evidence candidate frame-length estimation, scoring, and ranking for Stage 12.
"""

from typing import List, Dict, Optional
import numpy as np

from .models import PeriodicityCandidate, AutocorrPeak, SyncPatternResult
from .segmentation import (
    build_frame_matrix,
    compute_position_stability,
    find_optimal_frame_offset
)
from .repeated_patterns import analyze_repeated_prefixes


def estimate_frame_lengths(
    bits: np.ndarray,
    periodicity_candidates: List[PeriodicityCandidate],
    sync_results: List[SyncPatternResult] = None,
    candidate_sync_words: List[SyncPatternResult] = None,
    validation_hints: List[int] = None,
    standard_hints: List[int] = None,
    top_k: int = 5,
    min_length_bits: int = 32,
    max_length_bits: int = 4096
) -> List[PeriodicityCandidate]:
    """
    Synthesize multiple evidence sources to generate Top-K frame length hypotheses.
    """
    sync_results = sync_results or []
    candidate_sync_words = candidate_sync_words or []
    validation_hints = validation_hints or []
    standard_hints = standard_hints or [64, 128, 256, 512, 1024, 2048]

    candidate_dict: Dict[int, PeriodicityCandidate] = {}
    n = len(bits)

    # 1. Ingest periodicity candidates from autocorrelation
    for pc in periodicity_candidates:
        L = pc.period_bits
        if min_length_bits <= L <= max_length_bits:
            candidate_dict[L] = PeriodicityCandidate(
                period_bits=L,
                score=pc.score,
                support_sources=list(pc.support_sources),
                harmonic_relation=pc.harmonic_relation,
                estimated_frame_count=n // L if L > 0 else 0
            )

    # 2. Ingest sync spacing evidence only if periodic spacing is confident
    for sr in sync_results + candidate_sync_words:
        if sr.match_count >= 2 and sr.mean_spacing > 0 and sr.confidence >= 0.65 and sr.periodicity_score >= 0.50:
            int_sp = int(round(sr.mean_spacing))
            if min_length_bits <= int_sp <= max_length_bits:
                if int_sp in candidate_dict:
                    cand = candidate_dict[int_sp]
                    cand.score = float(np.clip(cand.score + 0.3 * sr.confidence, 0.0, 1.0))
                    if "sync_spacing" not in cand.support_sources:
                        cand.support_sources.append("sync_spacing")
                else:
                    candidate_dict[int_sp] = PeriodicityCandidate(
                        period_bits=int_sp,
                        score=float(0.6 * sr.confidence),
                        support_sources=["sync_spacing"],
                        harmonic_relation="sync_derived",
                        estimated_frame_count=n // int_sp
                    )

    # 3. Ingest validation hints
    for vh in validation_hints:
        if min_length_bits <= vh <= max_length_bits:
            if vh in candidate_dict:
                cand = candidate_dict[vh]
                cand.score = float(np.clip(cand.score + 0.25, 0.0, 1.0))
                if "validation_hint" not in cand.support_sources:
                    cand.support_sources.append("validation_hint")
            else:
                candidate_dict[vh] = PeriodicityCandidate(
                    period_bits=vh,
                    score=0.45,
                    support_sources=["validation_hint"],
                    harmonic_relation="hinted",
                    estimated_frame_count=n // vh
                )

    # 4. Deep structural evaluation of candidate lengths
    for L, cand in list(candidate_dict.items()):
        if n < 2 * L:
            continue

        best_off, stab_score = find_optimal_frame_offset(bits, frame_length=L, max_search_offset=min(64, L))
        cand.boundary_stability = stab_score

        prefixes = analyze_repeated_prefixes(bits, frame_length=L, prefix_lengths=[16, 24, 32])
        if prefixes:
            if "repeated_prefix" not in cand.support_sources:
                cand.support_sources.append("repeated_prefix")
            prefix_boost = 0.15 * (prefixes[0].occurrence_count / max(1.0, cand.estimated_frame_count))
            cand.score = float(np.clip(cand.score + prefix_boost + 0.2 * stab_score, 0.0, 1.0))
        else:
            cand.score = float(np.clip(cand.score + 0.15 * stab_score, 0.0, 1.0))

    candidates = list(candidate_dict.values())
    candidates.sort(key=lambda c: c.score, reverse=True)

    return candidates[:top_k]
