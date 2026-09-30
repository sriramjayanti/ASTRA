"""
models.py
Data models, enums, and dataclasses for ASTRA Stage 9 — FEC Candidate Testing Engine.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import hashlib
import json


class FECStatus(str, Enum):
    FEC_DECODED = "FEC_DECODED"
    FEC_PLAUSIBLE = "FEC_PLAUSIBLE"
    FEC_WEAK = "FEC_WEAK"
    FEC_FAILED = "FEC_FAILED"
    FEC_INVALID = "FEC_INVALID"


class FECFamily(str, Enum):
    NONE = "none"
    CONVOLUTIONAL = "convolutional"
    REED_SOLOMON = "reed_solomon"
    CONCATENATED = "concatenated"
    LDPC = "ldpc"


@dataclass
class FECProfile:
    """
    Standardized Forward Error Correction code profile specification.
    """
    profile_id: str
    family: str
    rate: float = 1.0
    rate_str: str = "1/1"
    # Convolutional parameters
    constraint_length: Optional[int] = None
    generators_octal: Optional[List[int]] = None
    termination: str = "terminated"
    puncturing_pattern: Optional[List[int]] = None
    # Reed-Solomon parameters
    n: Optional[int] = None
    k: Optional[int] = None
    symbol_bits: int = 8
    prim_poly: int = 0x11D
    fcr: int = 0
    shortened_n: Optional[int] = None
    shortened_k: Optional[int] = None
    # LDPC parameters
    matrix_type: Optional[str] = None
    block_size: Optional[int] = None
    # Concatenated parameters
    outer_profile: Optional[str] = None
    inner_profile: Optional[str] = None
    interleaving: Optional[str] = None
    # Metadata
    description: str = ""
    profile_hash: str = ""

    def __post_init__(self):
        if not self.profile_hash:
            rep = f"{self.profile_id}:{self.family}:{self.rate}:{self.constraint_length}:{self.generators_octal}:{self.n}:{self.k}:{self.outer_profile}:{self.inner_profile}"
            self.profile_hash = hashlib.sha256(rep.encode('utf-8')).hexdigest()[:16]


@dataclass
class DecoderResult:
    """Standardized outcome from an individual decoder backend execution."""
    success: bool
    decoded_bits: np.ndarray             # 1D uint8 array
    decoded_soft_info: Optional[np.ndarray] = None # 1D float32 array if available
    metrics: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    decoder_name: str = ""
    profile_id: str = ""
    runtime_ms: float = 0.0


@dataclass
class FECCandidateResult:
    """
    Result of testing a single FEC candidate hypothesis across the pipeline path.
    Preserves complete lineage from Modulation -> Baud -> Sync -> Demod -> Interleaver -> FEC.
    """
    candidate_id: str
    demod_variant_id: str
    interleaver_candidate_id: str
    fec_candidate_id: str
    path_id: str
    fec_family: str
    fec_parameters: Dict[str, Any]
    parent_candidate_id: str = ""
    decoder_type: str = ""
    profile_id: str = ""
    convergence_status: str = "CONVERGED"
    decoder_score: float = 0.0
    corrected_error_info: Dict[str, Any] = field(default_factory=dict)
    syndrome_metrics: Dict[str, Any] = field(default_factory=dict)
    path_metrics: Dict[str, Any] = field(default_factory=dict)
    lineage: List[Dict[str, Any]] = field(default_factory=list)
    input_bit_count: int = 0
    output_bit_count: int = 0
    decoded_hard_bits: Optional[np.ndarray] = None
    decoded_soft_information_if_available: Optional[np.ndarray] = None
    decoder_success: bool = True
    decoder_status: FECStatus = FECStatus.FEC_PLAUSIBLE
    decoder_metrics: Dict[str, Any] = field(default_factory=dict)
    code_rate: float = 1.0
    estimated_errors_corrected: int = 0
    iterations: int = 0
    syndrome_weight: float = 0.0
    path_metric: float = 0.0
    parity_check_success: bool = False
    structural_score: float = 0.0
    fec_quality_score: float = 0.0
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: Optional[str] = None

    @property
    def decoded_bits(self) -> np.ndarray:
        return self.decoded_hard_bits if self.decoded_hard_bits is not None else np.array([], dtype=np.uint8)

    def __post_init__(self):
        if not self.decoder_type:
            self.decoder_type = self.fec_family
        if not self.profile_id and "profile" in self.fec_parameters:
            self.profile_id = str(self.fec_parameters["profile"])
        if self.decoder_score == 0.0:
            self.decoder_score = self.fec_quality_score
        if not self.convergence_status:
            self.convergence_status = "CONVERGED" if self.decoder_success else "FAILED"
        if not self.corrected_error_info:
            self.corrected_error_info = {"errors_corrected": self.estimated_errors_corrected}
        if not self.syndrome_metrics:
            self.syndrome_metrics = {"syndrome_weight": self.syndrome_weight, "parity_check_success": self.parity_check_success}
        if not self.path_metrics:
            self.path_metrics = {"path_metric": self.path_metric, "iterations": self.iterations}

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        # Convert all metrics to native python types
        clean_metrics = {}
        for k, v in (self.decoder_metrics or {}).items():
            if isinstance(v, (bool, np.bool_)):
                clean_metrics[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                clean_metrics[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                clean_metrics[k] = float(v)
            else:
                clean_metrics[k] = v
                
        data = {
            "candidate_id": str(self.candidate_id),
            "parent_candidate_id": str(self.parent_candidate_id or self.candidate_id),
            "demod_variant_id": str(self.demod_variant_id),
            "interleaver_candidate_id": str(self.interleaver_candidate_id),
            "fec_candidate_id": str(self.fec_candidate_id),
            "path_id": str(self.path_id),
            "fec_family": str(self.fec_family),
            "decoder_type": str(self.decoder_type or self.fec_family),
            "profile_id": str(self.profile_id),
            "fec_parameters": self.fec_parameters,
            "convergence_status": str(self.convergence_status),
            "decoder_score": round(float(self.decoder_score or self.fec_quality_score), 4),
            "corrected_error_info": self.corrected_error_info,
            "syndrome_metrics": self.syndrome_metrics,
            "path_metrics": self.path_metrics,
            "lineage": self.lineage,
            "input_bit_count": int(self.input_bit_count),
            "output_bit_count": int(self.output_bit_count),
            "decoder_success": bool(self.decoder_success),
            "decoder_status": self.decoder_status.value if isinstance(self.decoder_status, FECStatus) else str(self.decoder_status),
            "decoder_metrics": clean_metrics,
            "code_rate": round(float(self.code_rate), 4),
            "estimated_errors_corrected": int(self.estimated_errors_corrected),
            "iterations": int(self.iterations),
            "syndrome_weight": round(float(self.syndrome_weight), 4),
            "path_metric": round(float(self.path_metric), 4),
            "parity_check_success": bool(self.parity_check_success),
            "structural_score": round(float(self.structural_score), 4),
            "fec_quality_score": round(float(self.fec_quality_score), 4),
            "rejected": bool(self.rejected),
            "rejection_reason": str(self.rejection_reason) if self.rejection_reason else None,
            "processing_history": self.processing_history,
        }
        if include_arrays and self.decoded_hard_bits is not None:
            data["decoded_hard_bits"] = self.decoded_hard_bits.tolist()
        return data


@dataclass
class FECTestResult:
    """
    Complete collection of tested FEC hypotheses across an input interleaver candidate stream.
    Passed downstream to the Stage 10 Validation Engine.
    """
    candidate_id: str
    demod_variant_id: str
    interleaver_candidate_id: str
    tested_candidates_count: int = 0
    surviving_candidates: List[FECCandidateResult] = field(default_factory=list)
    top_candidate: Optional[FECCandidateResult] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "demod_variant_id": self.demod_variant_id,
            "interleaver_candidate_id": self.interleaver_candidate_id,
            "tested_candidates_count": self.tested_candidates_count,
            "surviving_count": len(self.surviving_candidates),
            "top_candidate": self.top_candidate.to_dict(include_arrays) if self.top_candidate else None,
            "surviving_candidates": [c.to_dict(include_arrays) for c in self.surviving_candidates],
            "execution_time_ms": round(self.execution_time_ms, 2),
            "metadata": self.metadata,
        }
