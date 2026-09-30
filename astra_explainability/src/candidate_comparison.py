"""
candidate_comparison.py
Candidate comparison, score margin analysis, and ranking uncertainty evaluation.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .models import (
    CandidateExplanation,
    AstraStatus
)


class CandidateComparator:
    """Class wrapper for candidate comparisons, margin calculations, and ranking entropy."""

    def compare_candidates(
        self,
        candidate_list: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[CandidateExplanation]:
        """Compares top candidates and returns explanations list."""
        explanations, _, _ = compare_candidates(candidate_list, top_k=top_k)
        return explanations

    def calculate_ranking_entropy(self, scores: List[float]) -> float:
        """Computes normalized Shannon ranking entropy."""
        return compute_ranking_entropy(scores)

    def explain_why_lost(self, winner: Dict[str, Any], loser: Dict[str, Any]) -> str:
        """Explains why a candidate pipeline was rejected."""
        return explain_why_candidate_lost(winner, loser)


def compute_ranking_entropy(scores: List[float]) -> float:
    """
    Compute normalized Shannon entropy over candidate path scores.
    Low entropy: One candidate dominates decisively.
    High entropy: Competing candidates are close or tied.
    """
    if not scores or len(scores) <= 1:
        return 0.0

    arr = np.array(scores, dtype=np.float64)
    s_sum = np.sum(arr)
    if s_sum <= 0:
        return 1.0

    p = arr / s_sum
    p = p[p > 0]
    raw_entropy = -float(np.sum(p * np.log2(p)))
    max_entropy = np.log2(len(scores))
    if max_entropy <= 0:
        return 0.0
    return float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))


def explain_why_candidate_lost(
    winner: Dict[str, Any],
    loser: Dict[str, Any]
) -> str:
    """
    Generate a precise, human-readable reason why a candidate pipeline was rejected or ranked lower.
    """
    reasons = []

    # 1. CRC validation discrepancy
    w_crc = winner.get("crc_pass_ratio", 1.0 if winner.get("crc_passed") else 0.0)
    l_crc = loser.get("crc_pass_ratio", 1.0 if loser.get("crc_passed") else 0.0)
    if w_crc > l_crc:
        reasons.append(f"lower CRC verification rate ({l_crc*100:.0f}% vs {w_crc*100:.0f}% for winner)")

    # 2. FEC convergence discrepancy
    w_fec = winner.get("fec_match", winner.get("fec_reencoding_match_ratio", 1.0))
    l_fec = loser.get("fec_match", loser.get("fec_reencoding_match_ratio", 0.0))
    if w_fec is not None and l_fec is not None and w_fec - l_fec > 0.10:
        reasons.append(f"weaker FEC re-encoding match ({l_fec*100:.1f}% vs {w_fec*100:.1f}%)")

    # 3. Demodulation EVM discrepancy
    w_evm = winner.get("demod_evm_pct", 10.0)
    l_evm = loser.get("demod_evm_pct", 30.0)
    if l_evm - w_evm > 10.0:
        reasons.append(f"significantly higher demodulation EVM ({l_evm:.1f}% vs {w_evm:.1f}%)")

    # 4. Synchronization lock discrepancy
    w_lock = winner.get("timing_lock_metric", 0.9)
    l_lock = loser.get("timing_lock_metric", 0.5)
    if w_lock - l_lock > 0.20:
        reasons.append(f"unstable timing recovery lock ({l_lock:.2f} vs {w_lock:.2f})")

    # 5. Pipeline score difference
    w_score = winner.get("score", winner.get("pipeline_score", 0.9))
    l_score = loser.get("score", loser.get("pipeline_score", 0.4))
    if abs(w_score - l_score) < 0.01:
        reasons.append(f"Virtually tied pipeline score ({l_score:.3f} vs {w_score:.3f})")
    elif not reasons:
        reasons.append(f"lower overall pipeline ranking score ({l_score:.2f} vs {w_score:.2f})")

    p_id = loser.get("pipeline_id", "Candidate")
    return f"Ranked below winner ({p_id} lower pipeline score) due to: " + "; ".join(reasons) + "."


def compare_candidates(
    candidate_list: List[Dict[str, Any]],
    top_k: int = 3
) -> Tuple[List[CandidateExplanation], float, float]:
    """
    Compare top candidate pipelines, compute score margin, and assess ranking uncertainty.

    Returns:
        Tuple: (List of CandidateExplanation, score_margin, ranking_entropy)
    """
    if not candidate_list:
        return [], 0.0, 0.0

    # Sort descending by score
    sorted_cands = sorted(candidate_list, key=lambda c: float(c.get("score", c.get("pipeline_score", 0.0))), reverse=True)
    winner = sorted_cands[0]

    scores = [float(c.get("score", c.get("pipeline_score", 0.0))) for c in sorted_cands]
    margin = float(scores[0] - scores[1]) if len(scores) > 1 else 1.0
    entropy = compute_ranking_entropy(scores)

    explanations: List[CandidateExplanation] = []
    for rank_idx, cand in enumerate(sorted_cands[:top_k]):
        score = float(cand.get("score", cand.get("pipeline_score", 0.0)))
        cand_id = str(cand.get("pipeline_id", f"candidate_{rank_idx+1}"))
        c_margin = float(score - scores[rank_idx+1]) if rank_idx + 1 < len(scores) else 0.0

        # Assign status
        if rank_idx == 0:
            if score >= 0.85 and margin >= 0.20 and cand.get("crc_passed", True):
                status = AstraStatus.CONFIRMED
            elif score >= 0.60:
                status = AstraStatus.ESTIMATED
            else:
                status = AstraStatus.POSSIBLE
        else:
            status = AstraStatus.POSSIBLE if score >= 0.35 else AstraStatus.UNKNOWN

        # Strengths & Weaknesses
        strengths = []
        weaknesses = []

        if cand.get("crc_passed") or cand.get("crc_pass_ratio", 0.0) >= 0.80:
            strengths.append("High CRC verification success")
        elif cand.get("crc_passed") is False or cand.get("crc_pass_ratio", 1.0) == 0.0:
            weaknesses.append("Failed CRC checksum verification")

        fec_match = cand.get("fec_reencoding_match_ratio", cand.get("fec_match"))
        if fec_match is not None and fec_match >= 0.95:
            strengths.append(f"Strong FEC convergence ({fec_match*100:.1f}%)")
        elif fec_match is not None and fec_match < 0.80:
            weaknesses.append(f"Poor FEC re-encoding match ({fec_match*100:.1f}%)")

        evm = cand.get("demod_evm_pct")
        if evm is not None and evm < 15.0:
            strengths.append(f"Low demodulation EVM ({evm:.1f}%)")
        elif evm is not None and evm > 30.0:
            weaknesses.append(f"Elevated demodulation EVM ({evm:.1f}%)")

        why_lost = None if rank_idx == 0 else explain_why_candidate_lost(winner, cand)

        explanations.append(CandidateExplanation(
            pipeline_id=cand_id,
            rank=rank_idx + 1,
            score=score,
            status=status,
            margin_to_next=c_margin,
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            why_lost=why_lost,
            field_values={
                "modulation": cand.get("modulation"),
                "symbol_rate": cand.get("symbol_rate") or cand.get("baud_rate"),
                "interleaver": cand.get("interleaver_type"),
                "fec": cand.get("fec_type")
            }
        ))

    return explanations, margin, entropy
