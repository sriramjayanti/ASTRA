"""
explainability.py
Auditability and human-interpretable explanations for generated candidate hypotheses.
"""

from typing import Dict, Any, List
from .models import ReceiverHypothesis, CandidateSet


def explain_candidate(cand: ReceiverHypothesis) -> Dict[str, Any]:
    """
    Generate an auditable, human-readable breakdown of why a candidate exists,
    its evidence sources, and its suggested receiver route.
    """
    rf_str = f"{cand.rf_family_probability:.4f}" if cand.rf_family_probability is not None else "N/A"
    const_str = f"{cand.constellation_support_score:.4f}" if cand.constellation_support_score is not None else "N/A"

    explanation = {
        "candidate_id": cand.candidate_id,
        "rank": cand.candidate_rank,
        "hypothesis": f"{cand.modulation} @ {cand.symbol_rate_hz:g} Bd",
        "initial_score": round(cand.initial_score, 4),
        "status": cand.status,
        "samples_per_symbol": round(cand.samples_per_symbol, 3),
        "evidence_breakdown": {
            "fusion_modulation_prob": round(cand.modulation_probability, 4),
            "symbol_rate_score": round(cand.symbol_rate_score, 4),
            "rf_family_support": rf_str,
            "constellation_support": const_str,
        },
        "receiver_hints": {
            "family": cand.modulation_family,
            "demodulator_supported": cand.demodulator_supported,
            "demod_type": cand.demod_hints.get("demodulator_type", "unknown"),
            "sync_carrier": cand.sync_hints.get("carrier_recovery", "unspecified"),
            "sync_timing": cand.sync_hints.get("timing_recovery_methods", []),
        },
        "rejection": {
            "is_rejected": cand.rejected,
            "pruned_by_beam": cand.pruned_by_beam,
            "reason": cand.rejection_reason,
        }
    }
    return explanation


def format_candidate_summary_table(candidate_set: CandidateSet) -> str:
    """Format an ASCII / Markdown table for GUI / Hypothesis Panel display."""
    lines = []
    lines.append(f"{'Rank':<5} | {'Candidate ID':<26} | {'Modulation':<10} | {'Baud (Hz)':<10} | {'SPS':<8} | {'Score':<8} | {'Status':<12}")
    lines.append("-" * 92)
    
    for c in candidate_set.candidates:
        status_tag = c.status
        if c.rejected:
            status_tag = "REJECTED"
        elif c.pruned_by_beam:
            status_tag = "PRUNED"
            
        lines.append(
            f"#{c.candidate_rank:<4} | {c.candidate_id:<26} | {c.modulation:<10} | {c.symbol_rate_hz:<10.1f} | "
            f"{c.samples_per_symbol:<8.2f} | {c.initial_score:<8.4f} | {status_tag:<12}"
        )
    return "\n".join(lines)
