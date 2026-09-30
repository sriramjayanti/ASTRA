"""
periodicity.py
Periodicity analysis, harmonic clustering, subharmonic disambiguation, and period ranking.
"""

from typing import List, Dict
import numpy as np

from .models import AutocorrPeak, PeriodicityCandidate


def is_harmonic(lag1: int, lag2: int, tolerance: float = 0.05, max_harmonic: int = 6) -> bool:
    """
    Check if lag2 is an integer harmonic multiple of lag1 within tolerance (e.g. 2x, 3x, 4x).
    """
    if lag1 <= 0 or lag2 <= 0:
        return False

    ratio = lag2 / lag1
    nearest_int = round(ratio)
    if nearest_int < 2 or nearest_int > max_harmonic:
        return False

    rel_error = abs(ratio - nearest_int) / nearest_int
    return rel_error <= tolerance


def merge_harmonics(
    peaks: List[AutocorrPeak],
    tolerance: float = 0.05
) -> List[PeriodicityCandidate]:
    """
    Group autocorrelation peaks into harmonic families and identify fundamental candidates.
    """
    if not peaks:
        return []

    sorted_peaks = sorted(peaks, key=lambda p: p.lag)
    candidates: List[PeriodicityCandidate] = []
    seen_lags = set()

    for i, base_peak in enumerate(sorted_peaks):
        harmonic_peaks = [base_peak]

        # Check for higher harmonics
        for other_peak in sorted_peaks[i + 1:]:
            if is_harmonic(base_peak.lag, other_peak.lag, tolerance):
                harmonic_peaks.append(other_peak)

        has_harmonics = len(harmonic_peaks) > 1
        fundamental_strength = base_peak.correlation
        harmonic_boost = min(0.3, 0.08 * (len(harmonic_peaks) - 1)) if has_harmonics else 0.0
        combined_score = float(np.clip(fundamental_strength + harmonic_boost, 0.0, 1.0))

        support = ["autocorrelation_harmonic_family"] if has_harmonics else ["autocorrelation_fundamental"]

        candidates.append(PeriodicityCandidate(
            period_bits=base_peak.lag,
            score=combined_score,
            support_sources=support,
            harmonic_relation=f"family_size_{len(harmonic_peaks)}" if has_harmonics else "single_peak"
        ))

    # Sort candidates by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def rank_period_candidates(
    candidates: List[PeriodicityCandidate],
    sync_spacings: List[float] = None,
    validation_hints: List[int] = None,
    top_k: int = 10
) -> List[PeriodicityCandidate]:
    """
    Refine and re-rank periodicity candidates incorporating sync spacing and validation hints.
    """
    if not candidates:
        return []

    sync_spacings = sync_spacings or []
    validation_hints = validation_hints or []

    for cand in candidates:
        # Boost if supported by sync spacing
        for sp in sync_spacings:
            if abs(cand.period_bits - sp) / max(1.0, cand.period_bits) < 0.03:
                cand.score = float(np.clip(cand.score + 0.35, 0.0, 1.0))
                if "sync_spacing" not in cand.support_sources:
                    cand.support_sources.append("sync_spacing")

        # Boost if supported by upstream validation frame hints
        for vh in validation_hints:
            if abs(cand.period_bits - vh) / max(1.0, cand.period_bits) < 0.03:
                cand.score = float(np.clip(cand.score + 0.25, 0.0, 1.0))
                if "validation_hint" not in cand.support_sources:
                    cand.support_sources.append("validation_hint")

    # Final sort
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]
