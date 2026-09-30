"""
models.py
Data models and dataclasses for ASTRA Stage 5 — Candidate / Hypothesis Engine.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import json
from enum import Enum


class CandidateStatus(str, Enum):
    GENERATED = "GENERATED"
    SYNC_TESTING = "SYNC_TESTING"
    SYNC_FAILED = "SYNC_FAILED"
    SYNC_PASSED = "SYNC_PASSED"
    DEMOD_TESTING = "DEMOD_TESTING"
    DEMOD_FAILED = "DEMOD_FAILED"
    DEMOD_PASSED = "DEMOD_PASSED"
    INTERLEAVER_TESTING = "INTERLEAVER_TESTING"
    FEC_TESTING = "FEC_TESTING"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ReceiverHypothesis:
    """
    Represents a single receiver candidate hypothesis combining:
    Modulation Candidate × Symbol Rate Candidate + Supporting Evidence & Sync Hints.
    Memory-efficient: References signal metadata rather than copying raw IQ buffers.
    """
    candidate_id: str
    signal_id: str = "sig_unknown"

    # Core modulation & rate estimates
    modulation: str = "Unknown"
    modulation_probability: float = 0.0
    symbol_rate_hz: float = 0.0
    symbol_rate_score: float = 0.0
    samples_per_symbol: float = 0.0
    sample_rate_hz: float = 0.0

    # Categorization & Supporting evidence
    modulation_family: str = "UNKNOWN"
    rf_family_probability: Optional[float] = None
    constellation_support_score: Optional[float] = None

    # Ranking & Lifecycle
    initial_score: float = 0.0
    candidate_rank: int = 1
    status: str = CandidateStatus.GENERATED.value

    # Routing and downstream suggestions
    sync_hints: Dict[str, Any] = field(default_factory=dict)
    demod_hints: Dict[str, Any] = field(default_factory=dict)
    demodulator_supported: bool = True
    requires_manual_or_custom_path: bool = False

    # Pruning & Rejection
    pruned_by_beam: bool = False
    rejected: bool = False
    rejection_reason: Optional[str] = None

    # Immutable source evidence preservation
    source_evidence: Dict[str, Any] = field(default_factory=dict)

    # Auditable lifecycle history
    processing_history: List[Dict[str, Any]] = field(default_factory=list)

    # Placeholders for future pipeline stages (Stages 6 to 10)
    sync_result: Optional[Dict[str, Any]] = None
    demod_result: Optional[Dict[str, Any]] = None
    interleaver_result: Optional[Dict[str, Any]] = None
    fec_result: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    pipeline_score: Optional[float] = None

    def add_history_entry(self, stage: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Append an auditable event to the candidate's processing history."""
        entry = {
            "stage": stage,
            "status": status,
            "details": details or {}
        }
        self.processing_history.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReceiverHypothesis':
        return cls(**data)


@dataclass
class CandidateSet:
    """
    Container of generated, ranked, and pruned receiver hypotheses.
    """
    all_generated_count: int = 0
    valid_candidate_count: int = 0
    pruned_candidate_count: int = 0
    rejected_candidate_count: int = 0
    
    candidates: List[ReceiverHypothesis] = field(default_factory=list)
    top_candidates: List[ReceiverHypothesis] = field(default_factory=list)
    beam_candidates: List[ReceiverHypothesis] = field(default_factory=list)
    
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_generated_count": self.all_generated_count,
            "valid_candidate_count": self.valid_candidate_count,
            "pruned_candidate_count": self.pruned_candidate_count,
            "rejected_candidate_count": self.rejected_candidate_count,
            "candidates": [c.to_dict() for c in self.candidates],
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "beam_candidates": [c.to_dict() for c in self.beam_candidates],
            "generation_metadata": self.generation_metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def get_summary_lines(self) -> List[str]:
        """Compact GUI/terminal summary format."""
        lines = []
        for c in self.beam_candidates:
            pruned_tag = " [PRUNED]" if c.pruned_by_beam else ""
            rej_tag = f" [REJECTED: {c.rejection_reason}]" if c.rejected else ""
            lines.append(
                f"#{c.candidate_rank} {c.modulation} @ {c.symbol_rate_hz:g} Bd (SPS={c.samples_per_symbol:.2f}) - "
                f"Score: {c.initial_score:.4f} [Fam: {c.modulation_family}]{pruned_tag}{rej_tag}"
            )
        return lines
