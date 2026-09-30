"""
grid_generator.py
Cartesian product generator for Modulation Candidates × Symbol-Rate Candidates.
"""

import math
from typing import List, Dict, Any, Optional
from .models import ReceiverHypothesis, CandidateStatus
from .mappings import get_modulation_family, get_demod_hints, get_sync_hints
from .scoring import calculate_initial_score, normalize_support_score


def generate_candidate_grid(
    modulation_candidates: List[Dict[str, Any]],
    symbol_rate_candidates: List[Dict[str, Any]],
    sample_rate_hz: float,
    signal_id: str = "sig_001",
    rf_family_probs: Optional[Dict[str, float]] = None,
    constellation_support: Optional[Dict[str, float]] = None,
    constellation_stage: str = "pre_sync",
    config: Optional[Dict[str, Any]] = None,
    rate_source_tag: str = "symbol_rate_estimator"
) -> List[ReceiverHypothesis]:
    """
    Generate raw unpruned Candidate Hypotheses from the Cartesian product of:
    Modulation Candidates × Symbol Rate Candidates.
    
    Attaches modulation family, RF support, constellation support, demod/sync hints,
    calculates initial prior scores, and generates deterministic candidate IDs.
    """
    cfg = config or {}
    rate_variants_cfg = cfg.get("rate_variants", {})
    variants_enabled = rate_variants_cfg.get("enabled", False)
    offset_percentages = rate_variants_cfg.get("offsets_percent", [0.0]) if variants_enabled else [0.0]

    raw_hypotheses: List[ReceiverHypothesis] = []
    cand_index = 1

    for mod_item in modulation_candidates:
        mod_name = str(mod_item.get("modulation", "Unknown"))
        mod_prob = float(mod_item.get("probability", 0.0))
        mod_family = get_modulation_family(mod_name)
        
        # Look up optional Random Forest support for this modulation's family
        rf_fam_prob: Optional[float] = None
        if rf_family_probs:
            rf_fam_prob = rf_family_probs.get(mod_family, rf_family_probs.get(mod_name, None))
            rf_fam_prob = normalize_support_score(rf_fam_prob)

        # Look up optional Constellation support
        const_score: Optional[float] = None
        if constellation_support:
            const_score = constellation_support.get(mod_name, constellation_support.get(mod_family, None))
            const_score = normalize_support_score(const_score)

        demod_routing = get_demod_hints(mod_name)
        sync_routing = get_sync_hints(mod_name)
        is_unknown = (mod_family == "UNKNOWN" or mod_name.upper() in ["UNKNOWN", "NOISE"])

        for rate_item in symbol_rate_candidates:
            base_rate = float(rate_item.get("symbol_rate_hz", 0.0))
            rate_score = float(rate_item.get("score", 0.5))

            for offset_pct in offset_percentages:
                actual_rate = base_rate * (1.0 + offset_pct / 100.0) if not (math.isnan(base_rate) or math.isinf(base_rate)) else base_rate
                
                # Calculate initial prior score
                initial_score = calculate_initial_score(
                    p_mod=mod_prob,
                    p_rate=rate_score,
                    rf_family_prob=rf_fam_prob,
                    constellation_score=const_score,
                    config=cfg,
                    constellation_stage=constellation_stage
                )

                # Format clean deterministic candidate ID (safely handles non-finite values)
                sanitized_mod = mod_name.replace("-", "").replace("/", "").replace(" ", "_")
                if math.isnan(actual_rate) or math.isinf(actual_rate):
                    rate_str = "invalid"
                else:
                    rate_str = f"{int(round(actual_rate))}"
                    
                cand_id = f"cand_{sanitized_mod}_{rate_str}_{cand_index:03d}"
                cand_index += 1

                # Calculate non-integer samples per symbol
                if actual_rate > 0 and not (math.isnan(actual_rate) or math.isinf(actual_rate)):
                    sps = float(sample_rate_hz) / float(actual_rate)
                else:
                    sps = 0.0

                hypothesis = ReceiverHypothesis(
                    candidate_id=cand_id,
                    signal_id=signal_id,
                    modulation=mod_name,
                    modulation_probability=mod_prob if not (math.isnan(mod_prob) or math.isinf(mod_prob)) else 0.0,
                    symbol_rate_hz=actual_rate,
                    symbol_rate_score=rate_score if not (math.isnan(rate_score) or math.isinf(rate_score)) else 0.0,
                    samples_per_symbol=sps,
                    sample_rate_hz=float(sample_rate_hz),
                    modulation_family=mod_family,
                    rf_family_probability=rf_fam_prob,
                    constellation_support_score=const_score,
                    initial_score=initial_score,
                    status=CandidateStatus.GENERATED.value,
                    sync_hints=sync_routing,
                    demod_hints=demod_routing,
                    demodulator_supported=demod_routing.get("supported", True) and not is_unknown,
                    requires_manual_or_custom_path=is_unknown or not demod_routing.get("supported", True),
                    source_evidence={
                        "modulation_branch": mod_item,
                        "symbol_rate_branch": {
                            "base_rate_hz": base_rate,
                            "offset_pct": offset_pct,
                            "score": rate_score,
                            "source": rate_source_tag,
                        },
                        "rf_family_support": rf_fam_prob,
                        "constellation_support": const_score,
                        "constellation_stage": constellation_stage,
                    }
                )

                hypothesis.add_history_entry(
                    stage="generation",
                    status="GENERATED",
                    details={
                        "initial_score": initial_score,
                        "sps": sps,
                        "family": mod_family
                    }
                )

                raw_hypotheses.append(hypothesis)

    return raw_hypotheses
