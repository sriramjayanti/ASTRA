"""
scoring.py
Candidate prior scoring with optional Random Forest and Constellation support evidence.
"""

import math
from typing import Dict, Any, Optional
from .validators import sanitize_probability


def normalize_support_score(raw_score: Any) -> Optional[float]:
    """
    Normalize optional support score to [0.0, 1.0] range.
    Returns None if score is missing or invalid.
    """
    if raw_score is None:
        return None
    try:
        val = float(raw_score)
        if math.isnan(val) or math.isinf(val):
            return None
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return None


def calculate_initial_score(
    p_mod: float,
    p_rate: float,
    rf_family_prob: Optional[float] = None,
    constellation_score: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    constellation_stage: str = "pre_sync"
) -> float:
    """
    Calculate initial prior score combining modulation, baud, and optional auxiliary evidence.
    
    Modes:
      - 'weighted_product': score = (P_mod ^ a) * (P_rate ^ b) * (RF_fam ^ c) * (Const ^ d)
      - 'log_linear': log_score = a*log(P_mod) + b*log(P_rate) + c*log(RF_fam) + d*log(Const)
    """
    cfg = config or {}
    score_cfg = cfg.get("score", {})
    mode = score_cfg.get("mode", "weighted_product")
    eps = float(cfg.get("min_probability_epsilon", 1e-8))

    w_mod = float(score_cfg.get("modulation_weight", 1.0))
    w_rate = float(score_cfg.get("symbol_rate_weight", 1.0))
    w_rf = float(score_cfg.get("rf_family_weight", 0.25))
    w_const = float(score_cfg.get("constellation_weight", 0.25))

    # Stage-aware scaling for constellation evidence (pre-sync vs post-sync)
    if constellation_stage == "pre_sync":
        w_const *= float(score_cfg.get("pre_sync_constellation_weight_scale", 0.5))
    else:
        w_const *= float(score_cfg.get("post_sync_constellation_weight_scale", 1.0))

    # Sanitize inputs
    p_mod_clean = sanitize_probability(p_mod, default=eps, eps=eps)
    p_rate_clean = sanitize_probability(p_rate, default=eps, eps=eps)

    # Optional terms (default to 1.0 multiplier if missing so they don't penalize)
    rf_clean = sanitize_probability(rf_family_prob, default=None, eps=eps) if rf_family_prob is not None else None
    const_clean = sanitize_probability(constellation_score, default=None, eps=eps) if constellation_score is not None else None

    if mode == "log_linear":
        log_val = w_mod * math.log(p_mod_clean) + w_rate * math.log(p_rate_clean)
        if rf_clean is not None and w_rf > 0:
            log_val += w_rf * math.log(rf_clean)
        if const_clean is not None and w_const > 0:
            log_val += w_const * math.log(const_clean)
        
        # Exponentiate back to linear domain
        score = math.exp(log_val)
    else:
        # Default weighted product
        score = (p_mod_clean ** w_mod) * (p_rate_clean ** w_rate)
        if rf_clean is not None and w_rf > 0:
            score *= (rf_clean ** w_rf)
        if const_clean is not None and w_const > 0:
            score *= (const_clean ** w_const)

    if math.isnan(score) or math.isinf(score):
        return 0.0
    return float(max(0.0, min(1.0, score)))
