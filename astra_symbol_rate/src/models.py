"""
ASTRA Symbol-Rate (Baud) Estimation Engine Data Models and Data Classes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


class InvalidSignalError(Exception):
    """Raised when input IQ signal is empty, non-finite, or invalid."""
    pass


class NoCandidatesFoundError(Exception):
    """Raised when no plausible symbol rate candidates could be generated."""
    pass


class ModelNotFittedError(Exception):
    """Raised when attempting inference before the ranker model is fitted or loaded."""
    pass


@dataclass
class DSPRateEvidence:
    """
    Raw DSP measurements extracted across multiple independent estimation techniques.
    """
    autocorr_peaks: List[Dict[str, float]] = field(default_factory=list)
    power_autocorr_peaks: List[Dict[str, float]] = field(default_factory=list)
    if_peaks: List[Dict[str, float]] = field(default_factory=list)
    spectral_peak_spacings: List[Dict[str, float]] = field(default_factory=list)
    bandwidth_candidates: List[Dict[str, float]] = field(default_factory=list)
    cyclostationary_peaks: List[Dict[str, float]] = field(default_factory=list)
    occupied_bandwidth_hz: float = 0.0
    estimated_snr_db: float = 0.0
    cfo_estimate_hz: float = 0.0
    spectral_flatness: float = 0.0
    spectral_centroid_hz: float = 0.0
    clipping_ratio: float = 0.0
    sample_rate_hz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SymbolRateCandidate:
    """
    A single candidate symbol rate hypothesis with supporting DSP metrics and features.
    """
    rate_hz: float
    samples_per_symbol: float
    sources: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    supported_by: List[str] = field(default_factory=list)
    harmonic_ratio: Optional[float] = None
    raw_rate: Optional[float] = None
    snapped_rate: Optional[float] = None
    snap_distance_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rate_hz": float(self.rate_hz),
            "samples_per_symbol": float(self.samples_per_symbol),
            "score": float(self.score),
            "sources": self.sources,
            "supported_by": self.supported_by,
            "harmonic_ratio": self.harmonic_ratio,
            "raw_rate": self.raw_rate,
            "snapped_rate": self.snapped_rate,
            "snap_distance_percent": self.snap_distance_percent,
            "features": self.features,
        }


@dataclass
class SymbolRatePrediction:
    """
    Comprehensive symbol-rate prediction output designed to feed ASTRA's Candidate / Hypothesis Engine.
    """
    best_symbol_rate_hz: float
    samples_per_symbol: float
    confidence: float
    status: str  # "CONFIRMED", "ESTIMATED", "POSSIBLE", "UNKNOWN"
    confidence_margin: float
    top_k: List[Dict[str, Any]]
    dsp_evidence: Dict[str, Any]
    model_version: str = "astra_symbol_rate_xgb_v1.0"
    unknown_reason: Optional[str] = None
    modulation_context: Optional[Dict[str, Any]] = None

    @property
    def estimated_symbol_rate(self) -> float:
        return float(self.best_symbol_rate_hz)

    @property
    def primary_symbol_rate(self) -> float:
        return float(self.best_symbol_rate_hz)

    @property
    def symbol_rate_hz(self) -> float:
        return float(self.best_symbol_rate_hz)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "best_symbol_rate_hz": float(self.best_symbol_rate_hz),
            "primary_symbol_rate": float(self.best_symbol_rate_hz),
            "symbol_rate_hz": float(self.best_symbol_rate_hz),
            "samples_per_symbol": float(self.samples_per_symbol),
            "confidence": float(self.confidence),
            "status": self.status,
            "confidence_margin": float(self.confidence_margin),
            "unknown_reason": self.unknown_reason,
            "top_k": self.top_k,
            "model_version": self.model_version,
            "dsp_evidence": self.dsp_evidence,
            "modulation_context": self.modulation_context,
        }
        return d

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict() or hasattr(self, key)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

