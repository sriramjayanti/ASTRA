"""
ASTRA Multi-Branch Fusion Data Models and Exception Definitions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


class ClassMappingMismatchError(Exception):
    """Raised when the two branches have mismatched class alphabets or orders."""
    pass


class SourceAlignmentError(Exception):
    """Raised when predictions from different source IQ segments or windows are fused."""
    pass


class InvalidProbabilityError(Exception):
    """Raised when input probabilities contain NaN, Inf, negative values, or do not sum to 1."""
    pass


class InvalidWeightError(Exception):
    """Raised when fusion weights are negative or do not sum to 1."""
    pass


@dataclass
class CandidateItem:
    """Represents a single Top-K modulation candidate."""
    rank: int
    class_name: str
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "class": self.class_name,
            "probability": float(self.probability),
        }


@dataclass
class BranchPrediction:
    """
    Standardized prediction output from an individual classifier branch (1D or 2D).
    """
    model_name: str
    model_version: str
    class_names: List[str]
    logits: List[float]
    probabilities: List[float]
    predicted_class: str
    confidence: float
    top_k: List[Dict[str, Any]]
    feature_embedding: Optional[List[float]] = None
    source_signal_id: Optional[str] = None
    window_start: Optional[int] = None
    window_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class FusionPrediction:
    """
    Standardized comprehensive prediction output from the Multi-Branch Fusion Engine.
    """
    predicted_class: str
    confidence: float
    status: str  # "CONFIRMED", "ESTIMATED", "POSSIBLE", "UNKNOWN"
    branch_agreement: bool
    confidence_margin: float
    probability_entropy: float
    fusion_mode: str  # "weighted_probability" or "learned"
    top_k: List[Dict[str, Any]]
    probabilities: Dict[str, float]
    branch_evidence: Dict[str, Any]
    model_versions: Dict[str, str]
    unknown_reason: Optional[str] = None  # "explicit_unknown_class", "low_confidence", "high_uncertainty", "branch_conflict"
    fusion_weights: Optional[Dict[str, float]] = None
    logits: Optional[List[float]] = None
    fused_feature: Optional[List[float]] = None
    source_signal_id: Optional[str] = None
    window_start: Optional[int] = None
    window_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "predicted_class": self.predicted_class,
            "confidence": float(self.confidence),
            "status": self.status,
            "unknown_reason": self.unknown_reason,
            "branch_agreement": bool(self.branch_agreement),
            "confidence_margin": float(self.confidence_margin),
            "probability_entropy": float(self.probability_entropy),
            "fusion_mode": self.fusion_mode,
            "fusion_weights": self.fusion_weights,
            "top_k": self.top_k,
            "probabilities": self.probabilities,
            "branch_evidence": self.branch_evidence,
            "model_versions": self.model_versions,
            "source_signal_id": self.source_signal_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }
        if self.logits is not None:
            res["logits"] = self.logits
        if self.fused_feature is not None:
            res["fused_feature"] = self.fused_feature
        return res
