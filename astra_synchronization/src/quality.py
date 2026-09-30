"""
quality.py
Multi-dimensional lock quality assessment, constellation compactness scoring, and status evaluation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .models import SyncStatus


def calculate_constellation_compactness(symbols: np.ndarray) -> float:
    """
    Measure constellation clustering compactness from radial and phase spread.
    Returns normalized score in [0.0, 1.0] (higher = more structured constellation).
    """
    if symbols is None or len(symbols) < 4:
        return 0.0

    mags = np.abs(symbols)
    mean_mag = float(np.mean(mags)) + 1e-12
    std_mag = float(np.std(mags))
    mag_spread = std_mag / mean_mag

    compactness = float(1.0 / (1.0 + 2.0 * mag_spread))
    return float(max(0.0, min(1.0, compactness)))


def calculate_sync_lock_metrics(
    timing_lock: float,
    carrier_lock: float,
    frequency_lock: float,
    const_quality_before: float,
    const_quality_after: float
) -> Tuple[float, Dict[str, float]]:
    """
    Compute multi-dimensional lock scores and composite overall sync quality score.
    """
    t_lock = float(max(0.0, min(1.0, timing_lock)))
    c_lock = float(max(0.0, min(1.0, carrier_lock)))
    f_lock = float(max(0.0, min(1.0, frequency_lock)))
    
    improvement = max(0.0, const_quality_after - const_quality_before)
    const_score = float(min(1.0, const_quality_after + 0.3 * improvement))

    overall = (
        0.35 * t_lock +
        0.35 * c_lock +
        0.15 * f_lock +
        0.15 * const_score
    )
    overall_score = float(max(0.0, min(1.0, overall)))

    metrics = {
        "timing_lock_score": t_lock,
        "carrier_lock_score": c_lock,
        "frequency_lock_score": f_lock,
        "constellation_quality_score": const_score,
        "overall_sync_score": overall_score
    }
    return overall_score, metrics


def evaluate_sync_status(
    overall_score: float,
    timing_lock: float,
    carrier_lock: float,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[str, bool, Optional[str]]:
    """
    Determine whether candidate hypothesis achieved SYNC_PASSED, SYNC_WEAK, or SYNC_FAILED.
    """
    cfg = config or {}
    thresh_cfg = cfg.get("thresholds", {})
    
    th_pass = float(thresh_cfg.get("passed_overall_score", 0.50))
    th_timing = float(thresh_cfg.get("passed_timing_lock", 0.40))
    th_carrier = float(thresh_cfg.get("passed_carrier_lock", 0.40))
    th_weak = float(thresh_cfg.get("weak_overall_score", 0.30))

    if overall_score >= th_pass and timing_lock >= th_timing and carrier_lock >= th_carrier:
        return SyncStatus.SYNC_PASSED.value, True, None
    elif overall_score >= th_weak:
        reason = "weak_lock_convergence" if carrier_lock < th_carrier else "marginal_timing_lock"
        return SyncStatus.SYNC_WEAK.value, True, reason
    else:
        if timing_lock < th_timing:
            fail_reason = "timing_recovery_diverged_or_low_lock"
        elif carrier_lock < th_carrier:
            fail_reason = "carrier_phase_loop_unlocked"
        else:
            fail_reason = "overall_sync_score_below_threshold"
        return SyncStatus.SYNC_FAILED.value, False, fail_reason
