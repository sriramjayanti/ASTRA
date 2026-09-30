"""
models.py
Data models, enums, and dataclasses for ASTRA Stage 11 — Pipeline Scoring Model.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import json


class ConfidenceTier(str, Enum):
    CONFIRMED = "CONFIRMED"
    ESTIMATED = "ESTIMATED"
    POSSIBLE = "POSSIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class PipelinePathCandidate:
    """
    Complete candidate lineage and multi-stage telemetry.
    Carries evidence across Stage 5 -> 6 -> 7 -> 8 -> 9 -> 10.
    """
    signal_id: str
    candidate_path_id: str
    candidate_id: str
    demod_variant_id: str = "rot0"
    interleaver_candidate_id: str = "none"
    fec_candidate_id: str = "none"

    # Core Parameter Hypotheses
    modulation: str = "QPSK"
    symbol_rate_hz: float = 9600.0
    phase_variant: str = "rot0"
    interleaver_family: str = "none"
    fec_family: str = "none"

    # Upstream Stage Metrics & Result Dictionaries
    stage5_metrics: Dict[str, Any] = field(default_factory=dict)
    stage6_sync_metrics: Dict[str, Any] = field(default_factory=dict)
    stage7_demod_metrics: Dict[str, Any] = field(default_factory=dict)
    stage8_interleaver_metrics: Dict[str, Any] = field(default_factory=dict)
    stage9_fec_metrics: Dict[str, Any] = field(default_factory=dict)
    stage10_validation_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Decoded bits hash & lineage metadata
    decoded_bits_hash: str = ""
    processing_history: List[Dict[str, Any]] = field(default_factory=list)

    # Optional training label (strictly omitted from model input features)
    candidate_correct: Optional[int] = None


@dataclass
class RankedCandidate:
    """
    Individual candidate outcome within the signal-level ranking hierarchy.
    """
    rank: int
    pipeline_path_id: str
    candidate_id: str
    pipeline_score: float
    calibrated_probability: float
    modulation: str
    symbol_rate_hz: float
    phase_variant: str
    interleaver: str
    fec: str
    validation_status: str
    key_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "pipeline_path_id": str(self.pipeline_path_id),
            "candidate_id": str(self.candidate_id),
            "pipeline_score": round(float(self.pipeline_score), 4),
            "calibrated_probability": round(float(self.calibrated_probability), 4),
            "modulation": str(self.modulation),
            "symbol_rate_hz": float(self.symbol_rate_hz),
            "phase_variant": str(self.phase_variant),
            "interleaver": str(self.interleaver),
            "fec": str(self.fec),
            "validation_status": str(self.validation_status),
            "key_evidence": self.key_evidence,
        }


@dataclass
class PipelineRankingResult:
    """
    Final Stage 11 ranking result containing Top-K candidate paths for a given capture.
    """
    signal_id: str
    candidate_count: int
    ranked_candidates: List[RankedCandidate]
    top1_candidate_id: str
    top1_score: float
    top2_score: float
    score_margin: float
    ranking_uncertainty: float
    confidence_tier: ConfidenceTier = ConfidenceTier.POSSIBLE
    model_version: str = "astra_pipeline_xgb_v1"
    feature_schema_version: str = "pipeline_features_v1"
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": str(self.signal_id),
            "candidate_count": int(self.candidate_count),
            "top1_candidate_id": str(self.top1_candidate_id),
            "top1_score": round(float(self.top1_score), 4),
            "top2_score": round(float(self.top2_score), 4),
            "score_margin": round(float(self.score_margin), 4),
            "ranking_uncertainty": round(float(self.ranking_uncertainty), 4),
            "confidence_tier": self.confidence_tier.value,
            "model_version": str(self.model_version),
            "feature_schema_version": str(self.feature_schema_version),
            "fallback_used": bool(self.fallback_used),
            "ranked_candidates": [c.to_dict() for c in self.ranked_candidates],
        }


@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation metrics for Stage 11 ranker and classifier.
    """
    total_signals: int = 0
    total_candidates: int = 0
    positive_candidates: int = 0
    negative_candidates: int = 0
    candidate_recall_top1: float = 0.0
    candidate_recall_top3: float = 0.0
    candidate_recall_top5: float = 0.0
    mrr: float = 0.0
    conditional_top1_acc: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    ece: float = 0.0
    difficulty_breakdown: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_signals": int(self.total_signals),
            "total_candidates": int(self.total_candidates),
            "positive_candidates": int(self.positive_candidates),
            "negative_candidates": int(self.negative_candidates),
            "candidate_recall_top1": round(float(self.candidate_recall_top1), 4),
            "candidate_recall_top3": round(float(self.candidate_recall_top3), 4),
            "candidate_recall_top5": round(float(self.candidate_recall_top5), 4),
            "mrr": round(float(self.mrr), 4),
            "conditional_top1_acc": round(float(self.conditional_top1_acc), 4),
            "roc_auc": round(float(self.roc_auc), 4),
            "pr_auc": round(float(self.pr_auc), 4),
            "log_loss": round(float(self.log_loss), 4),
            "brier_score": round(float(self.brier_score), 4),
            "ece": round(float(self.ece), 4),
            "difficulty_breakdown": self.difficulty_breakdown,
            "feature_importance": self.feature_importance,
        }
