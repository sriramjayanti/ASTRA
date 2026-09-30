"""
ranking.py
Candidate ranking, top-K selection, margin calculation, and confidence tiering for ASTRA Stage 11.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .models import (
    ConfidenceTier,
    RankedCandidate,
    PipelineRankingResult,
    PipelinePathCandidate,
)
from .feature_importance import explain_candidate_prediction


def compute_ranking_uncertainty(scores: List[float]) -> float:
    """
    Compute normalized entropy uncertainty across top candidates.
    If scores are nearly tied (e.g. 0.33, 0.33, 0.33), uncertainty is high (~1.0).
    If one candidate dominates (e.g. 0.95, 0.03, 0.02), uncertainty is low (~0.0).
    """
    if len(scores) <= 1:
        return 0.0

    scores_arr = np.maximum(0.001, np.array(scores[:5], dtype=np.float64))
    probs = scores_arr / np.sum(scores_arr)
    entropy = - np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))
    return float(min(1.0, entropy / max(1e-6, max_entropy)))


def determine_confidence_tier(
    top1_score: float,
    score_margin: float,
    top1_candidate: RankedCandidate,
    min_confirmed: float = 0.80,
    min_estimated: float = 0.50,
    min_possible: float = 0.25,
) -> ConfidenceTier:
    """
    Assign global confidence tier: CONFIRMED / ESTIMATED / POSSIBLE / UNKNOWN.
    """
    val_status = top1_candidate.validation_status.upper()

    if (
        (top1_score >= min_confirmed and score_margin >= 0.15)
        or (val_status == "VALIDATION_STRONG" and top1_score >= 0.70)
    ):
        return ConfidenceTier.CONFIRMED

    if top1_score >= min_estimated and score_margin >= 0.05:
        return ConfidenceTier.ESTIMATED

    if top1_score >= min_possible:
        return ConfidenceTier.POSSIBLE

    return ConfidenceTier.UNKNOWN


def rank_pipeline_candidates(
    candidates: List[Any],
    scores: np.ndarray,
    calibrated_probs: Optional[np.ndarray] = None,
    candidate_features_list: Optional[List[Dict[str, float]]] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    top_k: int = 3,
    signal_id: str = "signal_0",
    model_version: str = "astra_pipeline_xgb_v1",
    fallback_used: bool = False,
) -> PipelineRankingResult:
    """
    Rank competing candidates for a single signal, build explanations, and return PipelineRankingResult.
    """
    if not candidates:
        return PipelineRankingResult(
            signal_id=signal_id,
            candidate_count=0,
            ranked_candidates=[],
            top1_candidate_id="",
            top1_score=0.0,
            top2_score=0.0,
            score_margin=0.0,
            ranking_uncertainty=1.0,
            confidence_tier=ConfidenceTier.UNKNOWN,
            model_version=model_version,
            fallback_used=fallback_used,
        )

    scores = np.asarray(scores, dtype=np.float64).ravel()
    calibrated_probs = np.asarray(calibrated_probs, dtype=np.float64).ravel() if calibrated_probs is not None else scores

    # Pair candidates with scores and sort descending
    paired = []
    for idx, cand in enumerate(candidates):
        sc = float(scores[idx]) if idx < len(scores) else 0.0
        prob = float(calibrated_probs[idx]) if idx < len(calibrated_probs) else sc
        feats = candidate_features_list[idx] if candidate_features_list and idx < len(candidate_features_list) else {}
        paired.append((sc, prob, cand, feats))

    paired.sort(key=lambda x: x[0], reverse=True)

    ranked_list: List[RankedCandidate] = []
    for rank_idx, (sc, prob, cand, feats) in enumerate(paired):
        # Extract metadata
        if hasattr(cand, "__dict__"):
            d = getattr(cand, "__dict__", {})
        elif isinstance(cand, dict):
            d = cand
        else:
            d = {}

        path_id = d.get("candidate_path_id", d.get("path_id", f"path_{rank_idx}"))
        cand_id = d.get("candidate_id", f"cand_{rank_idx}")
        mod = d.get("modulation", d.get("stage5_metrics", {}).get("modulation", "UNKNOWN"))
        baud = float(d.get("symbol_rate_hz", d.get("stage5_metrics", {}).get("symbol_rate_hz", 0.0)))
        phase = d.get("phase_variant", "rot0")
        inter = d.get("interleaver_family", d.get("stage8_interleaver_metrics", {}).get("interleaver_family", "none"))
        fec = d.get("fec_family", d.get("stage9_fec_metrics", {}).get("fec_family", "none"))
        val_status = str(d.get("validation_status", d.get("stage10_validation_metrics", {}).get("validation_status", "VALIDATION_INCONCLUSIVE")))

        key_evidence = explain_candidate_prediction(feats, feature_importance or {}) if feats else {}

        ranked_cand = RankedCandidate(
            rank=rank_idx + 1,
            pipeline_path_id=path_id,
            candidate_id=cand_id,
            pipeline_score=sc,
            calibrated_probability=prob,
            modulation=mod,
            symbol_rate_hz=baud,
            phase_variant=phase,
            interleaver=inter,
            fec=fec,
            validation_status=val_status,
            key_evidence=key_evidence,
        )
        ranked_list.append(ranked_cand)

    top1_score = ranked_list[0].pipeline_score if ranked_list else 0.0
    top2_score = ranked_list[1].pipeline_score if len(ranked_list) > 1 else 0.0
    margin = max(0.0, top1_score - top2_score)
    all_scores = [c.pipeline_score for c in ranked_list]
    uncertainty = compute_ranking_uncertainty(all_scores)

    tier = determine_confidence_tier(top1_score, margin, ranked_list[0]) if ranked_list else ConfidenceTier.UNKNOWN

    return PipelineRankingResult(
        signal_id=signal_id,
        candidate_count=len(ranked_list),
        ranked_candidates=ranked_list[:top_k],
        top1_candidate_id=ranked_list[0].candidate_id if ranked_list else "",
        top1_score=top1_score,
        top2_score=top2_score,
        score_margin=margin,
        ranking_uncertainty=uncertainty,
        confidence_tier=tier,
        model_version=model_version,
        fallback_used=fallback_used,
    )
