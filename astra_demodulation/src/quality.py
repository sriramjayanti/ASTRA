"""
quality.py
Demodulation quality telemetry, EVM integration, and status determination.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .models import DemodulationQuality, DemodStatus, HardDecisionResult, SoftDecisionResult
from .evm import compute_evm
from .noise import noise_var_to_snr_db


def evaluate_demodulation_quality(
    received_symbols: np.ndarray,
    hard_result: HardDecisionResult,
    soft_result: SoftDecisionResult,
    noise_var: float
) -> DemodulationQuality:
    """
    Compute comprehensive demodulation quality telemetry.
    """
    evm_rms, evm_pct, evm_db = compute_evm(
        received_symbols=received_symbols,
        reference_symbols=hard_result.nearest_points
    )

    snr_db = noise_var_to_snr_db(noise_var)
    mean_dist = float(np.mean(hard_result.decision_distances))
    mean_margin = float(np.mean(hard_result.decision_margins))
    mean_abs_llr = float(np.mean(np.abs(soft_result.llrs)))

    # Fraction of uncertain bits (|LLR| < 1.0)
    low_conf_fraction = float(np.mean(np.abs(soft_result.llrs) < 1.0))

    # Transparent rule-based composite quality score [0, 1]
    evm_factor = max(0.0, 1.0 - min(1.0, evm_rms))
    margin_factor = min(1.0, mean_margin / 0.5) if mean_margin > 0 else 0.5
    llr_factor = min(1.0, mean_abs_llr / 8.0)

    quality_score = float(0.40 * evm_factor + 0.30 * margin_factor + 0.30 * llr_factor)
    quality_score = max(0.0, min(1.0, quality_score))

    return DemodulationQuality(
        evm_rms=evm_rms,
        evm_percent=evm_pct,
        evm_db=evm_db,
        snr_estimate_db=snr_db,
        mean_decision_distance=mean_dist,
        mean_decision_margin=mean_margin,
        mean_abs_llr=mean_abs_llr,
        low_confidence_bit_fraction=low_conf_fraction,
        demodulation_quality_score=quality_score
    )


def determine_demod_status(
    quality: DemodulationQuality,
    symbol_count: int,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[str, bool, Optional[str]]:
    """
    Evaluate whether demodulation achieved DEMOD_PASSED, DEMOD_WEAK, or DEMOD_FAILED.
    """
    cfg = config or {}
    q_cfg = cfg.get("quality", {})

    min_symbols = int(q_cfg.get("minimum_symbols", 16))
    max_evm_pct = float(q_cfg.get("passed_evm_percent_max", 40.0))
    min_quality = float(q_cfg.get("passed_demod_quality_min", 0.45))
    min_weak = float(q_cfg.get("weak_demod_quality_min", 0.25))

    if symbol_count < min_symbols:
        return DemodStatus.DEMOD_FAILED.value, False, f"Insufficient symbols ({symbol_count} < {min_symbols})"

    if quality.demodulation_quality_score >= min_quality and quality.evm_percent <= max_evm_pct:
        return DemodStatus.DEMOD_PASSED.value, True, None
    elif quality.demodulation_quality_score >= min_weak:
        return DemodStatus.DEMOD_WEAK.value, True, "elevated_evm_or_marginal_llr"
    else:
        return DemodStatus.DEMOD_FAILED.value, False, "demodulation_quality_below_threshold"
