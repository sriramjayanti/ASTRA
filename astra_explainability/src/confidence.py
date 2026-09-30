"""
confidence.py
Decomposed, transparent confidence scoring engine across multi-stage pipeline evidence.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .models import (
    ConfidenceBreakdown,
    EvidenceItem,
    IndependenceGroup,
    EvidenceCategory,
    EvidenceStrength,
    Contradiction,
    MissingEvidence,
    FieldExplanation,
    CandidateExplanation
)


class ConfidenceEngine:
    """
    Computes transparent, auditable confidence scores across all pipeline dimensions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, status_engine: Optional[Any] = None):
        self.config = config or {}
        self.status_engine = status_engine
        w = self.config.get("confidence", self.config.get("confidence_weights", {}))
        self.weights = {
            "modulation": w.get("modulation_weight", 0.10),
            "symbol_rate": w.get("symbol_rate_weight", 0.10),
            "synchronization": w.get("synchronization_weight", 0.10),
            "demodulation": w.get("demodulation_weight", 0.10),
            "interleaver": w.get("interleaver_weight", 0.05),
            "fec": w.get("fec_weight", 0.15),
            "validation": w.get("validation_weight", 0.20),
            "pipeline_ranking": w.get("pipeline_ranking_weight", 0.10),
            "structure": w.get("structure_weight", 0.05),
            "payload": w.get("payload_weight", 0.05)
        }

    def compute_field_confidence(
        self,
        category: EvidenceCategory,
        evidence_items: List[EvidenceItem],
        default_score: float = 0.50
    ) -> float:
        """Calculate confidence for a single domain using its supporting evidence items."""
        cat_items = [it for it in evidence_items if it.category == category]
        if not cat_items:
            return default_score

        # Combine normalized scores favoring strong evidence
        scores = []
        for it in cat_items:
            weight = 1.5 if it.strength == EvidenceStrength.STRONG else (1.0 if it.strength == EvidenceStrength.MODERATE else 0.5)
            scores.append(it.normalized_score * weight)

        weights = [1.5 if it.strength == EvidenceStrength.STRONG else (1.0 if it.strength == EvidenceStrength.MODERATE else 0.5) for it in cat_items]
        combined = sum(scores) / max(1e-6, sum(weights))
        return float(np.clip(combined, 0.0, 1.0))

    def calculate_overall_confidence(
        self,
        field_explanations: Dict[str, FieldExplanation],
        candidate_explanations: List[CandidateExplanation],
        contradictions: List[Contradiction],
        missing_evidence: List[MissingEvidence],
        all_evidence: List[EvidenceItem]
    ) -> Tuple[float, ConfidenceBreakdown]:
        """
        Calculates overall ASTRA reasoning confidence and decomposes into components.
        """
        # 1. Field confidences
        c_mod = field_explanations.get("modulation", FieldExplanation("", None, None, 0.5)).confidence_score
        c_rate = field_explanations.get("symbol_rate", FieldExplanation("", None, None, 0.5)).confidence_score
        c_sync = field_explanations.get("synchronization", FieldExplanation("", None, None, 0.5)).confidence_score
        c_demod = field_explanations.get("demodulation", FieldExplanation("", None, None, 0.5)).confidence_score
        c_int = field_explanations.get("interleaver", FieldExplanation("", None, None, 0.5)).confidence_score
        c_fec = field_explanations.get("fec", FieldExplanation("", None, None, 0.5)).confidence_score
        c_val = field_explanations.get("validation", FieldExplanation("", None, None, 0.4)).confidence_score
        c_struct = field_explanations.get("frame_structure", FieldExplanation("", None, None, 0.5)).confidence_score
        c_pay = field_explanations.get("payload", FieldExplanation("", None, None, 0.5)).confidence_score

        # Pipeline ranking score
        top_cand = candidate_explanations[0] if candidate_explanations else None
        p_rank_score = top_cand.score if top_cand else 0.85
        score_margin = top_cand.margin_to_next if top_cand else 0.35

        # 2. Weighted Base Sum
        raw_weighted = (
            self.weights["modulation"] * c_mod +
            self.weights["symbol_rate"] * c_rate +
            self.weights["synchronization"] * c_sync +
            self.weights["demodulation"] * c_demod +
            self.weights["interleaver"] * c_int +
            self.weights["fec"] * c_fec +
            self.weights["validation"] * c_val +
            self.weights["pipeline_ranking"] * p_rank_score +
            self.weights["structure"] * c_struct +
            self.weights["payload"] * c_pay
        )

        # 3. Margin Adjustment
        m_factors = self.config.get("margin_factors", {})
        large_thresh = m_factors.get("large_margin_threshold", 0.30)
        small_thresh = m_factors.get("small_margin_threshold", 0.08)

        margin_adj = 0.0
        if score_margin >= large_thresh:
            margin_adj = m_factors.get("large_margin_bonus", 0.05)
        elif score_margin < small_thresh:
            margin_adj = -m_factors.get("small_margin_penalty", 0.10)

        # 4. Penalties
        contra_penalty = sum(
            c.penalty for c in contradictions if c.resolution_status != "RESOLVED_BY_DOWNSTREAM"
        )
        missing_penalty = sum(m.penalty for m in missing_evidence)

        # 5. Gating
        val_exp = field_explanations.get("validation")
        if val_exp and val_exp.confidence_score == 0.0:
            raw_weighted = min(raw_weighted, 0.58)

        overall = raw_weighted + margin_adj - contra_penalty - (missing_penalty * 0.5)
        overall = float(np.clip(overall, 0.05, 0.99))

        breakdown = ConfidenceBreakdown(
            overall_score=overall,
            calibrated_probability=None,
            modulation_confidence=c_mod,
            symbol_rate_confidence=c_rate,
            synchronization_confidence=c_sync,
            demodulation_confidence=c_demod,
            interleaver_confidence=c_int,
            fec_confidence=c_fec,
            validation_confidence=c_val,
            structure_confidence=c_struct,
            payload_confidence=c_pay,
            contradiction_penalty=contra_penalty,
            margin_adjustment=margin_adj
        )
        # Also store ranking_confidence as helper
        breakdown.ranking_confidence = p_rank_score + margin_adj
        breakdown.contradiction_penalties = contra_penalty

        return overall, breakdown

    def evaluate(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing_evidence: List[MissingEvidence],
        score_margin: float = 0.35
    ) -> ConfidenceBreakdown:
        """
        Synthesize multi-stage evidence into a comprehensive confidence breakdown.
        """
        # 1. Component Confidences
        c_mod = self.compute_field_confidence(EvidenceCategory.MODEL_PREDICTION, evidence_items, default_score=0.60)
        c_rate = self.compute_field_confidence(EvidenceCategory.DSP_MEASUREMENT, evidence_items, default_score=0.60)
        c_sync = self.compute_field_confidence(EvidenceCategory.SYNCHRONIZATION, evidence_items, default_score=0.50)
        c_demod = self.compute_field_confidence(EvidenceCategory.DEMODULATION, evidence_items, default_score=0.50)
        c_int = self.compute_field_confidence(EvidenceCategory.INTERLEAVER, evidence_items, default_score=0.50)
        c_fec = self.compute_field_confidence(EvidenceCategory.FEC, evidence_items, default_score=0.50)
        c_val = self.compute_field_confidence(EvidenceCategory.CRC_VALIDATION, evidence_items, default_score=0.40)
        c_struct = self.compute_field_confidence(EvidenceCategory.FRAME_STRUCTURE, evidence_items, default_score=0.50)
        c_pay = self.compute_field_confidence(EvidenceCategory.PAYLOAD_PARSE, evidence_items, default_score=0.50)

        # 2. Pipeline Scorer contribution
        p_rank_score = record.get("pipeline_score", 0.85)

        # 3. Weighted Base Sum
        raw_weighted = (
            self.weights["modulation"] * c_mod +
            self.weights["symbol_rate"] * c_rate +
            self.weights["synchronization"] * c_sync +
            self.weights["demodulation"] * c_demod +
            self.weights["interleaver"] * c_int +
            self.weights["fec"] * c_fec +
            self.weights["validation"] * c_val +
            self.weights["pipeline_ranking"] * p_rank_score +
            self.weights["structure"] * c_struct +
            self.weights["payload"] * c_pay
        )

        # 4. Margin Adjustment
        m_factors = self.config.get("margin_factors", {})
        large_thresh = m_factors.get("large_margin_threshold", 0.30)
        small_thresh = m_factors.get("small_margin_threshold", 0.08)

        margin_adj = 0.0
        if score_margin >= large_thresh:
            margin_adj = m_factors.get("large_margin_bonus", 0.05)
        elif score_margin < small_thresh:
            margin_adj = -m_factors.get("small_margin_penalty", 0.10)

        # 5. Penalties (Contradictions & Missing Evidence)
        contra_penalty = sum(
            c.penalty for c in contradictions if c.resolution_status != "RESOLVED_BY_DOWNSTREAM"
        )
        missing_penalty = sum(m.penalty for m in missing_evidence)

        # 6. Gating Logic
        crc_passed = record.get("crc_passed")
        crc_ratio = record.get("crc_pass_ratio")
        if (crc_passed is False) or (crc_ratio is not None and crc_ratio == 0.0):
            raw_weighted = min(raw_weighted, 0.58)

        # Final Overall Score
        overall = raw_weighted + margin_adj - contra_penalty - (missing_penalty * 0.5)
        overall = float(np.clip(overall, 0.05, 0.99))

        calibrated_prob = record.get("calibrated_pipeline_probability")

        breakdown = ConfidenceBreakdown(
            overall_score=overall,
            calibrated_probability=calibrated_prob,
            modulation_confidence=c_mod,
            symbol_rate_confidence=c_rate,
            synchronization_confidence=c_sync,
            demodulation_confidence=c_demod,
            interleaver_confidence=c_int,
            fec_confidence=c_fec,
            validation_confidence=c_val,
            structure_confidence=c_struct,
            payload_confidence=c_pay,
            contradiction_penalty=contra_penalty,
            margin_adjustment=margin_adj
        )
        breakdown.ranking_confidence = p_rank_score + margin_adj
        breakdown.contradiction_penalties = contra_penalty
        return breakdown
