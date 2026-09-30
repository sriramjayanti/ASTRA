"""
models.py
Data models, enums, and dataclasses for ASTRA Stage 10 — Validation Engine.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import hashlib
import json


class ValidationStatus(str, Enum):
    VALIDATION_STRONG = "VALIDATION_STRONG"
    VALIDATION_MODERATE = "VALIDATION_MODERATE"
    VALIDATION_WEAK = "VALIDATION_WEAK"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    VALIDATION_INCONCLUSIVE = "VALIDATION_INCONCLUSIVE"


class EvidenceCheckState(str, Enum):
    NOT_TESTED = "NOT_TESTED"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class CRCProfile:
    """CRC algorithm specification."""
    name: str
    width: int
    poly: int
    init: int
    refin: bool = False
    refout: bool = False
    xorout: int = 0
    check: Optional[int] = None
    description: str = ""


@dataclass
class CRCResult:
    """Outcome of CRC check across one or more candidate frames."""
    profile_name: str
    width: int
    checked_frames: int = 0
    passed_frames: int = 0
    pass_rate: float = 0.0
    calculated_crc: Optional[int] = None
    observed_crc: Optional[int] = None
    check_state: EvidenceCheckState = EvidenceCheckState.NOT_TESTED
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "width": self.width,
            "checked_frames": self.checked_frames,
            "passed_frames": self.passed_frames,
            "pass_rate": round(float(self.pass_rate), 4),
            "calculated_crc": hex(self.calculated_crc) if self.calculated_crc is not None else None,
            "observed_crc": hex(self.observed_crc) if self.observed_crc is not None else None,
            "check_state": self.check_state.value,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class ParityResult:
    """Outcome of simple/block parity consistency checks."""
    parity_type: str = "even"
    blocks_tested: int = 0
    blocks_passed: int = 0
    pass_rate: float = 0.0
    check_state: EvidenceCheckState = EvidenceCheckState.NOT_TESTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_type": self.parity_type,
            "blocks_tested": self.blocks_tested,
            "blocks_passed": self.blocks_passed,
            "pass_rate": round(float(self.pass_rate), 4),
            "check_state": self.check_state.value,
        }


@dataclass
class SyndromeResult:
    """Normalized FEC syndrome / parity telemetry from Stage 9."""
    syndrome_available: bool = False
    syndrome_valid: bool = False
    initial_syndrome_weight: float = 0.0
    final_syndrome_weight: float = 0.0
    normalized_syndrome_score: float = 0.0
    check_state: EvidenceCheckState = EvidenceCheckState.NOT_TESTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "syndrome_available": bool(self.syndrome_available),
            "syndrome_valid": bool(self.syndrome_valid),
            "initial_syndrome_weight": round(float(self.initial_syndrome_weight), 4),
            "final_syndrome_weight": round(float(self.final_syndrome_weight), 4),
            "normalized_syndrome_score": round(float(self.normalized_syndrome_score), 4),
            "check_state": self.check_state.value,
        }


@dataclass
class ReencodingResult:
    """Outcome of FEC re-encoding verification against received channel stream."""
    bits_compared: int = 0
    bit_match_fraction: float = 0.0
    weighted_match_score: float = 0.0
    check_state: EvidenceCheckState = EvidenceCheckState.NOT_TESTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bits_compared": self.bits_compared,
            "bit_match_fraction": round(float(self.bit_match_fraction), 4),
            "weighted_match_score": round(float(self.weighted_match_score), 4),
            "check_state": self.check_state.value,
        }


@dataclass
class FrameRepetitionResult:
    """Outcome of autocorrelation and inter-frame repetition analysis."""
    candidate_period_bits: int = 0
    repetition_count: int = 0
    mean_pairwise_similarity: float = 0.0
    periodicity_score: float = 0.0
    check_state: EvidenceCheckState = EvidenceCheckState.NOT_TESTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_period_bits": int(self.candidate_period_bits),
            "repetition_count": int(self.repetition_count),
            "mean_pairwise_similarity": round(float(self.mean_pairwise_similarity), 4),
            "periodicity_score": round(float(self.periodicity_score), 4),
            "check_state": self.check_state.value,
        }


@dataclass
class SyncWordResult:
    """Outcome of exact or Hamming-tolerant sync pattern detection."""
    pattern_id: str = ""
    match_count: int = 0
    positions: List[int] = field(default_factory=list)
    mean_hamming_distance: float = 0.0
    periodicity_score: float = 0.0
    check_state: EvidenceCheckState = EvidenceCheckState.NOT_TESTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "match_count": self.match_count,
            "positions": self.positions[:10],
            "mean_hamming_distance": round(float(self.mean_hamming_distance), 3),
            "periodicity_score": round(float(self.periodicity_score), 4),
            "check_state": self.check_state.value,
        }


@dataclass
class HeaderConsistencyResult:
    """Outcome of header field validation rules."""
    field_name: str = ""
    rule_type: str = ""
    frames_tested: int = 0
    frames_valid: int = 0
    consistency_score: float = 0.0
    check_state: EvidenceCheckState = EvidenceCheckState.NOT_TESTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "rule_type": self.rule_type,
            "frames_tested": self.frames_tested,
            "frames_valid": self.frames_valid,
            "consistency_score": round(float(self.consistency_score), 4),
            "check_state": self.check_state.value,
        }


@dataclass
class LengthConsistencyResult:
    """Outcome of header-declared length vs physical boundary consistency."""
    declared_length: int = 0
    observed_length: int = 0
    is_valid: bool = True
    check_state: EvidenceCheckState = EvidenceCheckState.NOT_TESTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "declared_length": int(self.declared_length),
            "observed_length": int(self.observed_length),
            "is_valid": bool(self.is_valid),
            "check_state": self.check_state.value,
        }


@dataclass
class StructuralResult:
    """Supporting structural bitstream telemetry."""
    binary_entropy: float = 1.0
    bit_balance_zero: float = 0.5
    mean_run_length: float = 1.5
    byte_alignment_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "binary_entropy": round(float(self.binary_entropy), 4),
            "bit_balance_zero": round(float(self.bit_balance_zero), 4),
            "mean_run_length": round(float(self.mean_run_length), 3),
            "byte_alignment_score": round(float(self.byte_alignment_score), 4),
        }


@dataclass
class ValidationResult:
    """
    Comprehensive multi-source validation result for a complete candidate pipeline path.
    Preserves all evidence separately for Stage 11 XGBoost scoring.
    """
    pipeline_path_id: str
    candidate_id: str
    demod_variant_id: str
    interleaver_candidate_id: str
    fec_candidate_id: str
    parent_candidate_id: str = ""
    lineage: List[Dict[str, Any]] = field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.VALIDATION_INCONCLUSIVE
    overall_validation_score: float = 0.0
    crc_results: List[CRCResult] = field(default_factory=list)
    parity_results: List[ParityResult] = field(default_factory=list)
    syndrome_result: SyndromeResult = field(default_factory=SyndromeResult)
    reencoding_result: ReencodingResult = field(default_factory=ReencodingResult)
    frame_repetition_result: FrameRepetitionResult = field(default_factory=FrameRepetitionResult)
    sync_word_results: List[SyncWordResult] = field(default_factory=list)
    header_results: List[HeaderConsistencyResult] = field(default_factory=list)
    length_result: LengthConsistencyResult = field(default_factory=LengthConsistencyResult)
    structural_result: StructuralResult = field(default_factory=StructuralResult)
    evidence_count: int = 0
    strong_evidence_count: int = 0
    contradiction_count: int = 0
    contradiction_details: List[str] = field(default_factory=list)
    validation_features: Dict[str, float] = field(default_factory=dict)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_reason: Optional[str] = None
    decoded_bits_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_path_id": str(self.pipeline_path_id),
            "candidate_id": str(self.candidate_id),
            "parent_candidate_id": str(self.parent_candidate_id or self.candidate_id),
            "demod_variant_id": str(self.demod_variant_id),
            "interleaver_candidate_id": str(self.interleaver_candidate_id),
            "fec_candidate_id": str(self.fec_candidate_id),
            "lineage": self.lineage,
            "validation_status": self.validation_status.value,
            "overall_validation_score": round(float(self.overall_validation_score), 4),
            "evidence_count": int(self.evidence_count),
            "strong_evidence_count": int(self.strong_evidence_count),
            "contradiction_count": int(self.contradiction_count),
            "contradiction_details": self.contradiction_details,
            "decoded_bits_hash": self.decoded_bits_hash,
            "crc_results": [r.to_dict() for r in self.crc_results],
            "parity_results": [r.to_dict() for r in self.parity_results],
            "syndrome_result": self.syndrome_result.to_dict(),
            "reencoding_result": self.reencoding_result.to_dict(),
            "frame_repetition_result": self.frame_repetition_result.to_dict(),
            "sync_word_results": [r.to_dict() for r in self.sync_word_results],
            "header_results": [r.to_dict() for r in self.header_results],
            "length_result": self.length_result.to_dict(),
            "structural_result": self.structural_result.to_dict(),
            "validation_features": self.validation_features,
            "processing_history": self.processing_history,
            "failure_reason": str(self.failure_reason) if self.failure_reason else None,
        }
