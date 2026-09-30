"""
models.py
Data models, enums, and dataclasses for ASTRA Stage 12 — Bitstream Intelligence Engine.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import json


class StructureStatus(str, Enum):
    STRUCTURE_STRONG = "STRUCTURE_STRONG"
    STRUCTURE_MODERATE = "STRUCTURE_MODERATE"
    STRUCTURE_WEAK = "STRUCTURE_WEAK"
    STRUCTURE_UNKNOWN = "STRUCTURE_UNKNOWN"


@dataclass
class AutocorrPeak:
    """Individual peak from bipolar bitstream autocorrelation."""
    lag: int
    correlation: float
    prominence: float = 0.0
    harmonic_group: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lag": int(self.lag),
            "correlation": round(float(self.correlation), 4),
            "prominence": round(float(self.prominence), 4),
            "harmonic_group": int(self.harmonic_group),
        }


@dataclass
class PeriodicityCandidate:
    """Ranked frame period hypothesis."""
    period_bits: int
    score: float
    support_sources: List[str] = field(default_factory=list)
    harmonic_relation: str = "fundamental"
    estimated_frame_count: int = 0
    boundary_stability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_bits": int(self.period_bits),
            "score": round(float(self.score), 4),
            "support_sources": self.support_sources,
            "harmonic_relation": self.harmonic_relation,
            "estimated_frame_count": int(self.estimated_frame_count),
            "boundary_stability": round(float(self.boundary_stability), 4),
        }


@dataclass
class SyncPatternResult:
    """Exact or Hamming-tolerant sync pattern detection and spacing."""
    pattern_id: str
    bit_pattern: List[int] = field(default_factory=list)
    length: int = 0
    positions: List[int] = field(default_factory=list)
    match_scores: List[float] = field(default_factory=list)
    mean_spacing: float = 0.0
    spacing_variance: float = 0.0
    match_count: int = 0
    periodicity_score: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": str(self.pattern_id),
            "length": int(self.length),
            "positions": self.positions[:10],
            "match_count": int(self.match_count),
            "mean_spacing": round(float(self.mean_spacing), 2),
            "spacing_variance": round(float(self.spacing_variance), 2),
            "periodicity_score": round(float(self.periodicity_score), 4),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class RepeatedPattern:
    """Frequent recurring bit sequence discovery."""
    pattern_bits: str
    length: int
    occurrence_count: int
    positions: List[int] = field(default_factory=list)
    mean_spacing: float = 0.0
    spacing_variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_bits": self.pattern_bits,
            "length": int(self.length),
            "occurrence_count": int(self.occurrence_count),
            "positions": self.positions[:10],
            "mean_spacing": round(float(self.mean_spacing), 2),
            "spacing_variance": round(float(self.spacing_variance), 2),
        }


@dataclass
class RunLengthStats:
    """Run length metrics for 0s and 1s."""
    mean_run_length: float = 1.0
    median_run_length: float = 1.0
    max_run_length: int = 1
    run_length_variance: float = 0.0
    long_run_fraction: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_run_length": round(float(self.mean_run_length), 3),
            "median_run_length": round(float(self.median_run_length), 3),
            "max_run_length": int(self.max_run_length),
            "run_length_variance": round(float(self.run_length_variance), 3),
            "long_run_fraction": round(float(self.long_run_fraction), 4),
        }


@dataclass
class BitBalance:
    """Bit balance telemetry."""
    zero_fraction: float = 0.5
    one_fraction: float = 0.5
    imbalance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zero_fraction": round(float(self.zero_fraction), 4),
            "one_fraction": round(float(self.one_fraction), 4),
            "imbalance": round(float(self.imbalance), 4),
        }


@dataclass
class ByteAlignmentResult:
    """Evaluation of 8 bit offsets for byte-level structure."""
    best_offset: int = 0
    offset_entropies: List[float] = field(default_factory=list)
    zero_byte_fractions: List[float] = field(default_factory=list)
    printable_fractions: List[float] = field(default_factory=list)
    structural_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_offset": int(self.best_offset),
            "offset_entropies": [round(float(e), 4) for e in self.offset_entropies],
            "zero_byte_fractions": [round(float(z), 4) for z in self.zero_byte_fractions],
            "printable_fractions": [round(float(p), 4) for p in self.printable_fractions],
            "structural_scores": [round(float(s), 4) for s in self.structural_scores],
        }


@dataclass
class StructuralRegion:
    """Identified structural region within candidate frame (e.g. fixed header, variable payload)."""
    start_bit: int
    end_bit: int
    region_type: str = "variable" # fixed, semi_fixed, variable, checksum, unknown
    mean_stability: float = 0.5
    mean_entropy: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_bit": int(self.start_bit),
            "end_bit": int(self.end_bit),
            "region_type": str(self.region_type),
            "mean_stability": round(float(self.mean_stability), 4),
            "mean_entropy": round(float(self.mean_entropy), 4),
        }


@dataclass
class BitstreamIntelligenceResult:
    """
    Comprehensive structural analysis outcome for a recovered candidate bitstream.
    Feeds structural feature maps and candidate frame lengths to Stage 13 CNN + Transformer.
    """
    pipeline_path_id: str
    bit_count: int
    status: StructureStatus = StructureStatus.STRUCTURE_UNKNOWN

    # Global Statistics
    bit_balance: BitBalance = field(default_factory=BitBalance)
    binary_entropy_global: float = 1.0
    ngram_entropies: Dict[str, float] = field(default_factory=dict)
    run_length_stats: RunLengthStats = field(default_factory=RunLengthStats)

    # Periodic & Autocorrelation Telemetry
    autocorrelation_peaks: List[AutocorrPeak] = field(default_factory=list)
    periodicity_candidates: List[PeriodicityCandidate] = field(default_factory=list)
    frame_length_candidates: List[PeriodicityCandidate] = field(default_factory=list)

    # Sync & Pattern Discovery
    sync_results: List[SyncPatternResult] = field(default_factory=list)
    candidate_sync_words: List[SyncPatternResult] = field(default_factory=list)
    repeated_patterns: List[RepeatedPattern] = field(default_factory=list)

    # Byte & Regional Frame Structure
    byte_alignment: ByteAlignmentResult = field(default_factory=ByteAlignmentResult)
    structural_regions: List[StructuralRegion] = field(default_factory=list)
    position_stability_map: List[float] = field(default_factory=list)
    position_entropy_map: List[float] = field(default_factory=list)

    # Sliding-Window Telemetry
    entropy_windows: List[float] = field(default_factory=list)
    entropy_change_points: List[int] = field(default_factory=list)

    # Stage 13 Neural Representation Maps & Features
    global_features: Dict[str, float] = field(default_factory=dict)
    sequence_feature_map: Optional[np.ndarray] = None # Shape: [N_bits, N_channels]
    
    # Lineage & History
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    decoded_bits_hash: str = ""

    def to_dict(self, include_sequence_maps: bool = False) -> Dict[str, Any]:
        data = {
            "pipeline_path_id": str(self.pipeline_path_id),
            "bit_count": int(self.bit_count),
            "status": self.status.value,
            "binary_entropy_global": round(float(self.binary_entropy_global), 4),
            "bit_balance": self.bit_balance.to_dict(),
            "ngram_entropies": {k: round(float(v), 4) for k, v in self.ngram_entropies.items()},
            "run_length_stats": self.run_length_stats.to_dict(),
            "autocorrelation_peaks": [p.to_dict() for p in self.autocorrelation_peaks[:10]],
            "periodicity_candidates": [p.to_dict() for p in self.periodicity_candidates[:5]],
            "frame_length_candidates": [f.to_dict() for f in self.frame_length_candidates[:5]],
            "sync_results": [s.to_dict() for s in self.sync_results],
            "candidate_sync_words": [s.to_dict() for s in self.candidate_sync_words[:5]],
            "repeated_patterns": [r.to_dict() for r in self.repeated_patterns[:5]],
            "byte_alignment": self.byte_alignment.to_dict(),
            "structural_regions": [r.to_dict() for r in self.structural_regions],
            "position_stability_sample": [round(float(x), 3) for x in self.position_stability_map[:32]],
            "position_entropy_sample": [round(float(x), 3) for x in self.position_entropy_map[:32]],
            "entropy_change_points": self.entropy_change_points[:10],
            "global_features": self.global_features,
            "decoded_bits_hash": str(self.decoded_bits_hash),
            "processing_history": self.processing_history,
        }
        if include_sequence_maps and self.sequence_feature_map is not None:
            data["sequence_feature_map_shape"] = list(self.sequence_feature_map.shape)
        return data
