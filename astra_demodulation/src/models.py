"""
models.py
Data models and dataclasses for ASTRA Stage 7 — Demodulation Engine.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import json


class DemodStatus(str, Enum):
    DEMOD_PASSED = "DEMOD_PASSED"
    DEMOD_WEAK = "DEMOD_WEAK"
    DEMOD_FAILED = "DEMOD_FAILED"


@dataclass
class ConstellationDefinition:
    """
    Canonical definition of a digital modulation constellation.
    Shared consistently across transmitter and receiver demappers.
    """
    name: str
    complex_points: np.ndarray  # Shape: [M], dtype: complex64
    bit_labels: np.ndarray      # Shape: [M, bits_per_symbol], dtype: uint8
    symbol_indices: np.ndarray  # Shape: [M], dtype: int32
    average_energy: float = 1.0
    bits_per_symbol: int = 1


@dataclass
class HardDecisionResult:
    """Hard decision symbol indices and sliced bitstream."""
    symbol_indices: np.ndarray     # Shape: [N], integer symbol index [0, M-1]
    hard_bits: np.ndarray          # Shape: [N * bits_per_symbol], dtype: uint8 (0 or 1)
    nearest_points: np.ndarray     # Shape: [N], complex reference points
    decision_distances: np.ndarray # Shape: [N], Euclidean distance to nearest reference
    decision_margins: np.ndarray   # Shape: [N], distance margin (dist_2nd - dist_1st)


@dataclass
class SoftDecisionResult:
    """Soft bit Log-Likelihood Ratios (LLRs) and bit-level confidences."""
    llrs: np.ndarray               # Shape: [N * bits_per_symbol], dtype: float32
    bit_confidences: np.ndarray    # Shape: [N * bits_per_symbol], normalized confidence
    symbol_confidences: np.ndarray # Shape: [N], per-symbol confidence
    llr_mode: str = "max_log"
    noise_variance: float = 0.05


@dataclass
class DemodulationQuality:
    """Multi-dimensional telemetry quantifying demodulation confidence and EVM."""
    evm_rms: float = 0.0
    evm_percent: float = 0.0
    evm_db: float = 0.0
    snr_estimate_db: float = 0.0
    mean_decision_distance: float = 0.0
    mean_decision_margin: float = 0.0
    mean_abs_llr: float = 0.0
    low_confidence_bit_fraction: float = 0.0
    demodulation_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "evm_rms": round(self.evm_rms, 4),
            "evm_percent": round(self.evm_percent, 2),
            "evm_db": round(self.evm_db, 2),
            "snr_estimate_db": round(self.snr_estimate_db, 2),
            "mean_decision_distance": round(self.mean_decision_distance, 4),
            "mean_decision_margin": round(self.mean_decision_margin, 4),
            "mean_abs_llr": round(self.mean_abs_llr, 3),
            "low_confidence_bit_fraction": round(self.low_confidence_bit_fraction, 4),
            "demodulation_quality_score": round(self.demodulation_quality_score, 4),
        }


@dataclass
class DemodulationVariant:
    """
    A single phase-ambiguity / tone-swap bitstream orientation candidate.
    Passed downstream to Stage 8/9/10 for Interleaving, FEC, and Validation.
    """
    candidate_id: str = ""
    parent_candidate_id: str = ""
    modulation: str = "Unknown"
    baud: float = 0.0
    sample_rate: float = 0.0
    hard_bits: Optional[np.ndarray] = None  # uint8 array
    soft_llrs: Optional[np.ndarray] = None  # float32 array
    bit_count: int = 0
    symbol_count: int = 0
    phase_variant: str = "0.0_deg"
    sync_score: float = 0.0
    timing_score: float = 0.0
    carrier_score: float = 0.0
    residual_cfo: float = 0.0
    evm: float = 0.0
    llr_confidence: float = 0.0
    demod_quality: float = 0.0
    lineage: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy & compatibility fields
    variant_id: str = ""
    ambiguity_type: str = "phase_rotation"
    rotation_deg: float = 0.0
    quality: DemodulationQuality = field(default_factory=DemodulationQuality)

    def __post_init__(self):
        if not self.candidate_id and self.variant_id:
            self.candidate_id = self.variant_id
        elif not self.variant_id and self.candidate_id:
            self.variant_id = self.candidate_id
        if not self.phase_variant:
            self.phase_variant = f"{self.rotation_deg:.1f}_deg"
        if self.hard_bits is not None and self.bit_count == 0:
            self.bit_count = len(self.hard_bits)
        if self.quality and self.demod_quality == 0.0:
            self.demod_quality = self.quality.demodulation_quality_score
        if self.quality and self.evm == 0.0:
            self.evm = self.quality.evm_percent
        if self.soft_llrs is not None and self.llr_confidence == 0.0:
            self.llr_confidence = float(np.mean(np.abs(self.soft_llrs))) if len(self.soft_llrs) > 0 else 0.0

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        data = {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "modulation": self.modulation,
            "baud": round(float(self.baud), 2),
            "sample_rate": round(float(self.sample_rate), 2),
            "bit_count": int(self.bit_count),
            "symbol_count": int(self.symbol_count),
            "phase_variant": str(self.phase_variant),
            "sync_score": round(float(self.sync_score), 4),
            "timing_score": round(float(self.timing_score), 4),
            "carrier_score": round(float(self.carrier_score), 4),
            "residual_cfo": round(float(self.residual_cfo), 2),
            "evm": round(float(self.evm), 2),
            "llr_confidence": round(float(self.llr_confidence), 4),
            "demod_quality": round(float(self.demod_quality), 4),
            "lineage": self.lineage,
            "metadata": self.metadata,
            "variant_id": self.variant_id,
            "ambiguity_type": self.ambiguity_type,
            "rotation_deg": self.rotation_deg,
            "quality": self.quality.to_dict() if self.quality else {},
        }
        if include_arrays:
            data["hard_bits"] = self.hard_bits.tolist() if self.hard_bits is not None else []
            data["soft_llrs"] = [round(float(v), 3) for v in self.soft_llrs] if self.soft_llrs is not None else []
        return data


@dataclass
class DemodulationResult:
    """
    Primary canonical output of the Demodulation Engine containing nominal hard/soft bitstreams
    and all phase-ambiguity candidate variants.
    """
    candidate_id: str
    parent_candidate_id: str = ""
    modulation: str = "Unknown"
    modulation_family: str = "UNKNOWN"
    baud: float = 0.0
    sample_rate: float = 0.0
    hard_bits: Optional[np.ndarray] = None
    soft_llrs: Optional[np.ndarray] = None
    bit_count: int = 0
    symbol_count: int = 0
    phase_variant: str = "nominal"
    sync_score: float = 0.0
    timing_score: float = 0.0
    carrier_score: float = 0.0
    residual_cfo: float = 0.0
    evm: float = 0.0
    llr_confidence: float = 0.0
    demod_quality: float = 0.0
    lineage: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    success: bool = False
    status: str = DemodStatus.DEMOD_FAILED.value
    bits_per_symbol: int = 1

    # Nominal Demodulated Streams
    hard_symbol_indices: Optional[np.ndarray] = None

    # Rotational / Permutation Ambiguity Candidates
    phase_variants: List[DemodulationVariant] = field(default_factory=list)

    # Telemetry and Diagnostics
    quality: DemodulationQuality = field(default_factory=DemodulationQuality)
    noise_variance: float = 0.05
    noise_source: str = "decision_directed"

    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_reason: Optional[str] = None

    def __post_init__(self):
        if self.hard_bits is not None and self.bit_count == 0:
            self.bit_count = len(self.hard_bits)
        if self.quality and self.demod_quality == 0.0:
            self.demod_quality = self.quality.demodulation_quality_score
        if self.quality and self.evm == 0.0:
            self.evm = self.quality.evm_percent
        if self.soft_llrs is not None and self.llr_confidence == 0.0:
            self.llr_confidence = float(np.mean(np.abs(self.soft_llrs))) if len(self.soft_llrs) > 0 else 0.0

    def add_history_entry(self, stage: str, status: str, details: Optional[Dict[str, Any]] = None):
        self.processing_history.append({
            "stage": stage,
            "status": status,
            "details": details or {}
        })
        self.lineage.append({
            "stage": stage,
            "status": status,
            "details": details or {}
        })

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        data = {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "modulation": self.modulation,
            "modulation_family": self.modulation_family,
            "baud": round(float(self.baud), 2),
            "sample_rate": round(float(self.sample_rate), 2),
            "bit_count": int(self.bit_count),
            "symbol_count": int(self.symbol_count),
            "phase_variant": self.phase_variant,
            "sync_score": round(float(self.sync_score), 4),
            "timing_score": round(float(self.timing_score), 4),
            "carrier_score": round(float(self.carrier_score), 4),
            "residual_cfo": round(float(self.residual_cfo), 2),
            "evm": round(float(self.evm), 2),
            "llr_confidence": round(float(self.llr_confidence), 4),
            "demod_quality": round(float(self.demod_quality), 4),
            "lineage": self.lineage,
            "metadata": self.metadata,
            "success": self.success,
            "status": self.status,
            "bits_per_symbol": self.bits_per_symbol,
            "noise_variance": round(float(self.noise_variance), 6),
            "noise_source": self.noise_source,
            "quality": self.quality.to_dict() if self.quality else {},
            "phase_variants": [v.to_dict(include_arrays=include_arrays) for v in self.phase_variants],
            "failure_reason": self.failure_reason,
            "processing_history": self.processing_history,
        }
        if include_arrays:
            data["hard_bits"] = self.hard_bits.tolist() if self.hard_bits is not None else []
            data["soft_llrs"] = [round(float(v), 3) for v in self.soft_llrs] if self.soft_llrs is not None else []
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_arrays=False), indent=indent)

