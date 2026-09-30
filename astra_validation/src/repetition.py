"""
repetition.py
Frame repetition and periodicity analysis for ASTRA Stage 10.
Detects periodic frame boundaries using autocorrelation and pairwise Hamming similarity.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from .models import FrameRepetitionResult, EvidenceCheckState


def estimate_repetition_periods(
    bits: Union[np.ndarray, List[int]],
    min_period: int = 32,
    max_period: int = 2048,
    max_candidates: int = 16,
    threshold: float = 0.25,
) -> List[int]:
    """
    Estimate candidate frame periods via bitstream autocorrelation.
    """
    bits = np.asarray(bits, dtype=np.float32).ravel()
    n = len(bits)
    if n < min_period * 2:
        return []

    # Center around zero (+1, -1)
    b_centered = 2.0 * bits - 1.0

    # FFT-based autocorrelation
    n_fft = 1 << (2 * n - 1).bit_length()
    f_b = np.fft.rfft(b_centered, n=n_fft)
    autocorr = np.fft.irfft(f_b * np.conj(f_b), n=n_fft)[:n]
    
    # Normalize
    if autocorr[0] > 0:
        autocorr /= autocorr[0]

    candidate_lags = []
    max_p = min(max_period, n // 2)

    # Find local peaks in [min_period, max_p]
    for lag in range(min_period, max_p - 1):
        if autocorr[lag] > threshold:
            if autocorr[lag] >= autocorr[lag - 1] and autocorr[lag] >= autocorr[lag + 1]:
                candidate_lags.append((float(autocorr[lag]), lag))

    # Sort descending by peak height
    candidate_lags.sort(key=lambda x: x[0], reverse=True)
    return [lag for _, lag in candidate_lags[:max_candidates]]


def segment_by_period(
    bits: Union[np.ndarray, List[int]],
    period: int,
) -> List[np.ndarray]:
    """
    Segment a bitstream into consecutive frames of fixed length `period`.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    num_frames = len(bits) // period
    frames = []
    for i in range(num_frames):
        frames.append(bits[i * period:(i + 1) * period])
    return frames


def score_frame_repetition(
    bits: Union[np.ndarray, List[int]],
    candidate_periods: Optional[List[int]] = None,
    min_period: int = 32,
    max_period: int = 2048,
) -> FrameRepetitionResult:
    """
    Analyze frame repetition across candidate periods.
    Measures mean pairwise similarity across segmented frames.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    total_bits = len(bits)

    if total_bits < min_period * 2:
        return FrameRepetitionResult(check_state=EvidenceCheckState.NOT_TESTED)

    if candidate_periods is not None and len(candidate_periods) > 0:
        all_candidates = candidate_periods
    else:
        auto_periods = estimate_repetition_periods(
            bits,
            min_period=min_period,
            max_period=max_period,
            threshold=0.15,
        )
        standard_hints = [64, 128, 256, 512, 1024, 2048]
        all_candidates = list(dict.fromkeys(auto_periods + [p for p in standard_hints if min_period <= p <= max_period and p * 2 <= total_bits]))

    best_period = 0
    best_similarity = 0.0
    best_count = 0
    best_score = 0.0
    best_rank_metric = 0.0

    for period in all_candidates:
        frames = segment_by_period(bits, period)
        if len(frames) < 2:
            continue

        num_compare = min(len(frames), 10)
        pairwise_sims = []
        for i in range(num_compare - 1):
            f1 = frames[i]
            f2 = frames[i + 1]
            match_frac = float(np.mean(f1 == f2))
            pairwise_sims.append(match_frac)

        if not pairwise_sims:
            continue

        mean_sim = float(np.mean(pairwise_sims))
        periodicity_score = max(0.0, (mean_sim - 0.50) * 2.0)
        
        # Rank by periodicity score with repetition count confidence
        rank_metric = periodicity_score * min(1.0, len(frames) / 3.0)
        if rank_metric > best_rank_metric or (abs(rank_metric - best_rank_metric) < 1e-4 and len(frames) > best_count):
            best_rank_metric = rank_metric
            best_score = periodicity_score
            best_period = period
            best_similarity = mean_sim
            best_count = len(frames)

    check_state = EvidenceCheckState.NOT_TESTED
    if best_score >= 0.40 and best_count >= 3:
        check_state = EvidenceCheckState.PASS
    elif best_score >= 0.20:
        check_state = EvidenceCheckState.PASS
    elif best_count >= 4 and best_score < 0.05:
        check_state = EvidenceCheckState.NOT_TESTED

    return FrameRepetitionResult(
        candidate_period_bits=best_period,
        repetition_count=best_count,
        mean_pairwise_similarity=best_similarity,
        periodicity_score=best_score,
        check_state=check_state,
    )
