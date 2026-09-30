"""
models.py
Data models and dataclasses for ASTRA Stage 8 — Interleaver Candidate Testing Engine.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import hashlib
import json


class InterleaverStatus(str, Enum):
    INTERLEAVER_GENERATED = "INTERLEAVER_GENERATED"
    INTERLEAVER_PLAUSIBLE = "INTERLEAVER_PLAUSIBLE"
    INTERLEAVER_WEAK = "INTERLEAVER_WEAK"
    INTERLEAVER_INVALID = "INTERLEAVER_INVALID"
    INTERLEAVER_REJECTED = "INTERLEAVER_REJECTED"


class InterleaverFamily(str, Enum):
    IDENTITY = "identity"
    BLOCK = "block"
    CONVOLUTIONAL = "convolutional"
    HELICAL = "helical"
    PSEUDO_RANDOM = "pseudo_random"


@dataclass
class PermutationMapping:
    """
    Generic finite block permutation abstraction.
    Guarantees inverse[forward[i]] == i for all i in [0, length-1].
    """
    forward_indices: np.ndarray    # Array where output[i] = input[forward_indices[i]]
    inverse_indices: np.ndarray    # Array where restored[i] = interleaved[inverse_indices[i]]
    length: int
    family: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    mapping_hash: str = ""

    def __post_init__(self):
        if not self.mapping_hash and len(self.inverse_indices) > 0:
            # Hash inverse mapping indices for quick duplicate collision detection
            self.mapping_hash = hashlib.sha256(self.inverse_indices.tobytes()).hexdigest()[:16]

    def is_valid(self) -> bool:
        """Verify permutation contains every index [0, length-1] exactly once."""
        if len(self.inverse_indices) != self.length:
            return False
        if len(np.unique(self.inverse_indices)) != self.length:
            return False
        if np.min(self.inverse_indices) != 0 or np.max(self.inverse_indices) != self.length - 1:
            return False
        return True


@dataclass
class ConvolutionalDeinterleaverState:
    """
    State buffer for stateful convolutional deinterleavers across streaming chunks.
    For branch i (0 to B-1), the deinterleaver delay line has (B - 1 - i) * delay_step registers.
    """
    branch_count: int
    delay_step: int
    shift_registers: List[List[Union[int, float]]] = field(default_factory=list)
    commutator_pos: int = 0
    total_processed: int = 0

    def __post_init__(self):
        if not self.shift_registers:
            self.reset()

    def reset(self):
        """Initialize FIFO delay lines with zeros."""
        self.shift_registers = []
        for i in range(self.branch_count):
            delay_len = (self.branch_count - 1 - i) * self.delay_step
            self.shift_registers.append([0.0] * delay_len)
        self.commutator_pos = 0
        self.total_processed = 0


@dataclass
class StructuralFeatures:
    """Multi-dimensional structural telemetry extracted from candidate bitstreams."""
    binary_entropy: float = 1.0
    bit_balance_zero_fraction: float = 0.5
    bit_balance_one_fraction: float = 0.5
    mean_run_length: float = 2.0
    max_run_length: int = 1
    run_length_entropy: float = 0.0
    autocorrelation_peak: float = 0.0
    peak_lag: int = 0
    periodicity_score: float = 0.0
    remainder_fraction: float = 0.0
    mean_abs_llr: float = 0.0
    low_confidence_fraction: float = 0.0
    known_sync_correlation: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "binary_entropy": round(self.binary_entropy, 4),
            "bit_balance_zero_fraction": round(self.bit_balance_zero_fraction, 4),
            "bit_balance_one_fraction": round(self.bit_balance_one_fraction, 4),
            "mean_run_length": round(self.mean_run_length, 3),
            "max_run_length": int(self.max_run_length),
            "run_length_entropy": round(self.run_length_entropy, 4),
            "autocorrelation_peak": round(self.autocorrelation_peak, 4),
            "peak_lag": int(self.peak_lag),
            "periodicity_score": round(self.periodicity_score, 4),
            "remainder_fraction": round(self.remainder_fraction, 4),
            "mean_abs_llr": round(self.mean_abs_llr, 3),
            "low_confidence_fraction": round(self.low_confidence_fraction, 4),
            "known_sync_correlation": round(self.known_sync_correlation, 4),
        }


@dataclass
class InterleaverCandidateResult:
    """
    Result of a single tested deinterleaver candidate hypothesis.
    Passed to Stage 9 FEC testing.
    """
    candidate_id: str
    demod_variant_id: str
    interleaver_candidate_id: str
    interleaver_family: str
    parameters: Dict[str, Any]
    parent_candidate_id: str = ""
    interleaver_type: str = ""
    score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    lineage: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    status: InterleaverStatus = InterleaverStatus.INTERLEAVER_PLAUSIBLE
    deinterleaved_hard_bits: Optional[np.ndarray] = None  # uint8 array
    deinterleaved_soft_llrs: Optional[np.ndarray] = None  # float32 array
    structural_score: float = 0.0
    periodicity_score: float = 0.0
    entropy_score: float = 0.0
    autocorrelation_score: float = 0.0
    run_length_score: float = 0.0
    candidate_prior: float = 1.0
    overall_interleaver_score: float = 0.0
    structural_features: StructuralFeatures = field(default_factory=StructuralFeatures)
    permutation_hash: str = ""
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: Optional[str] = None
    lazy_materialized: bool = False
    permutation_mapping: Optional[PermutationMapping] = None
    state: Optional[ConvolutionalDeinterleaverState] = None

    def __post_init__(self):
        if not self.interleaver_type:
            self.interleaver_type = self.interleaver_family
        if self.score == 0.0:
            self.score = self.overall_interleaver_score
        if not self.evidence and self.structural_features:
            self.evidence = self.structural_features.to_dict()

    def materialize_arrays(self, source_hard: np.ndarray, source_soft: Optional[np.ndarray] = None):
        """Materialize bits and LLRs on-demand if lazy evaluation is enabled."""
        if self.deinterleaved_hard_bits is not None:
            return
        if self.permutation_mapping is not None and self.permutation_mapping.inverse_indices is not None:
            inv = self.permutation_mapping.inverse_indices
            L = len(inv)
            self.deinterleaved_hard_bits = source_hard[:L][inv].astype(np.uint8)
            if source_soft is not None:
                self.deinterleaved_soft_llrs = source_soft[:L][inv].astype(np.float32)
            self.lazy_materialized = False

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        data = {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "demod_variant_id": self.demod_variant_id,
            "interleaver_candidate_id": self.interleaver_candidate_id,
            "interleaver_family": self.interleaver_family,
            "interleaver_type": self.interleaver_type or self.interleaver_family,
            "parameters": self.parameters,
            "score": round(float(self.score or self.overall_interleaver_score), 4),
            "evidence": self.evidence or (self.structural_features.to_dict() if self.structural_features else {}),
            "lineage": self.lineage,
            "success": self.success,
            "status": self.status.value if isinstance(self.status, InterleaverStatus) else str(self.status),
            "structural_score": round(self.structural_score, 4),
            "periodicity_score": round(self.periodicity_score, 4),
            "entropy_score": round(self.entropy_score, 4),
            "autocorrelation_score": round(self.autocorrelation_score, 4),
            "run_length_score": round(self.run_length_score, 4),
            "candidate_prior": round(self.candidate_prior, 3),
            "overall_interleaver_score": round(self.overall_interleaver_score, 4),
            "structural_features": self.structural_features.to_dict() if self.structural_features else {},
            "permutation_hash": self.permutation_hash,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "lazy_materialized": self.lazy_materialized,
            "processing_history": self.processing_history,
        }
        if include_arrays:
            if self.deinterleaved_hard_bits is not None:
                data["deinterleaved_hard_bits"] = self.deinterleaved_hard_bits.tolist()
            if self.deinterleaved_soft_llrs is not None:
                data["deinterleaved_soft_llrs"] = self.deinterleaved_soft_llrs.tolist()
        else:
            data["bit_count"] = len(self.deinterleaved_hard_bits) if self.deinterleaved_hard_bits is not None else 0
            data["llr_count"] = len(self.deinterleaved_soft_llrs) if self.deinterleaved_soft_llrs is not None else 0
        return data


@dataclass
class InterleaverTestResult:
    """
    Comprehensive result of testing all interleaver candidates across demodulation variants.
    """
    candidate_id: str
    demod_variant_id: str
    tested_candidates_count: int = 0
    surviving_candidates: List[InterleaverCandidateResult] = field(default_factory=list)
    top_candidate: Optional[InterleaverCandidateResult] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "demod_variant_id": self.demod_variant_id,
            "tested_candidates_count": self.tested_candidates_count,
            "surviving_count": len(self.surviving_candidates),
            "top_candidate": self.top_candidate.to_dict(include_arrays) if self.top_candidate else None,
            "surviving_candidates": [c.to_dict(include_arrays) for c in self.surviving_candidates],
            "execution_time_ms": round(self.execution_time_ms, 2),
            "metadata": self.metadata,
        }
