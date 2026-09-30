"""
ASTRA Constellation Analysis Engine Data Models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class InvalidIQError(Exception):
    """Raised when input IQ samples are empty, non-finite, or invalid."""
    pass


class ClusteringFailedError(Exception):
    """Raised when clustering algorithms fail to converge or extract geometry."""
    pass


@dataclass
class ConstellationEvidence:
    """
    Spatial, geometric, and clustering evidence extracted from the IQ constellation plane.
    """
    dbscan_cluster_count: int = 0
    dbscan_noise_ratio: float = 0.0
    kmeans_silhouette_scores: Dict[int, float] = field(default_factory=dict)
    kmeans_inertias: Dict[int, float] = field(default_factory=dict)
    radial_ring_count: int = 0
    radial_ring_amplitudes: List[float] = field(default_factory=list)
    constant_modulus_variance: float = 0.0
    angular_phase_uniformity: float = 0.0
    square_grid_compactness: float = 0.0
    estimated_evm_percent: float = 0.0
    estimated_snr_db: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstellationCandidate:
    """
    A single constellation type hypothesis (e.g. QPSK, 16-QAM).
    """
    constellation_type: str
    m_ary_order: int
    score: float
    supported_by: List[str] = field(default_factory=list)
    evm_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constellation_type": self.constellation_type,
            "m_ary_order": self.m_ary_order,
            "score": float(round(self.score, 4)),
            "supported_by": self.supported_by,
            "evm_percent": float(round(self.evm_percent, 2)) if self.evm_percent is not None else None,
        }


@dataclass
class ConstellationPrediction:
    """
    Unified constellation analysis output feeding ASTRA's Candidate / Hypothesis Engine.
    """
    best_constellation: str
    m_ary_order: int
    confidence: float
    status: str  # "CONFIRMED", "ESTIMATED", "POSSIBLE", "UNKNOWN"
    top_k: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    model_version: str = "astra_constellation_v1.0"
    unknown_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_constellation": self.best_constellation,
            "m_ary_order": self.m_ary_order,
            "confidence": float(round(self.confidence, 4)),
            "status": self.status,
            "top_k": self.top_k,
            "model_version": self.model_version,
            "unknown_reason": self.unknown_reason,
            "evidence": self.evidence,
        }
