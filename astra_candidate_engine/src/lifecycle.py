"""
lifecycle.py
Candidate lifecycle state management, rejection tracking, and downstream expansion hooks.
"""

from typing import Dict, Any, Optional, List
from .models import ReceiverHypothesis, CandidateStatus


def update_candidate_status(
    candidate: ReceiverHypothesis,
    new_status: str,
    stage_name: str,
    details: Optional[Dict[str, Any]] = None
) -> ReceiverHypothesis:
    """Transition candidate into a new status and record in history."""
    candidate.status = new_status
    candidate.add_history_entry(stage=stage_name, status=new_status, details=details or {})
    return candidate


def record_sync_result(
    candidate: ReceiverHypothesis,
    passed: bool,
    sync_metrics: Dict[str, Any]
) -> ReceiverHypothesis:
    """Record synchronization outcome (Stage 6 hook)."""
    candidate.sync_result = sync_metrics
    new_status = CandidateStatus.SYNC_PASSED.value if passed else CandidateStatus.SYNC_FAILED.value
    if not passed:
        candidate.rejected = True
        candidate.rejection_reason = sync_metrics.get("rejection_reason", "sync_failed")
    return update_candidate_status(candidate, new_status, "synchronization", sync_metrics)


def record_demod_result(
    candidate: ReceiverHypothesis,
    passed: bool,
    demod_metrics: Dict[str, Any]
) -> ReceiverHypothesis:
    """Record demodulation outcome (Stage 7 hook)."""
    candidate.demod_result = demod_metrics
    new_status = CandidateStatus.DEMOD_PASSED.value if passed else CandidateStatus.DEMOD_FAILED.value
    if not passed:
        candidate.rejected = True
        candidate.rejection_reason = demod_metrics.get("rejection_reason", "demod_failed")
    return update_candidate_status(candidate, new_status, "demodulation", demod_metrics)


def record_validation_result(
    candidate: ReceiverHypothesis,
    passed: bool,
    validation_metrics: Dict[str, Any],
    pipeline_score: Optional[float] = None
) -> ReceiverHypothesis:
    """Record end-to-end validation outcome (Stage 10 hook)."""
    candidate.validation_result = validation_metrics
    candidate.pipeline_score = pipeline_score
    new_status = CandidateStatus.VALIDATED.value if passed else CandidateStatus.REJECTED.value
    if not passed:
        candidate.rejected = True
        candidate.rejection_reason = validation_metrics.get("rejection_reason", "validation_failed")
    return update_candidate_status(candidate, new_status, "validation", validation_metrics)


def expand_with_interleavers(
    candidate: ReceiverHypothesis,
    interleaver_hypotheses: List[Dict[str, Any]]
) -> List[ReceiverHypothesis]:
    """
    Hook for Stage 8 Interleaver candidate expansion.
    Leaves candidate intact and returns expansion stubs for downstream testing.
    """
    expanded = []
    for idx, il_cfg in enumerate(interleaver_hypotheses):
        child = ReceiverHypothesis.from_dict(candidate.to_dict())
        child.candidate_id = f"{candidate.candidate_id}_il{idx:02d}"
        child.status = CandidateStatus.INTERLEAVER_TESTING.value
        child.interleaver_result = {"suggested_interleaver": il_cfg}
        child.add_history_entry("interleaver_expansion", "EXPANDED", il_cfg)
        expanded.append(child)
    return expanded


def expand_with_fec(
    candidate: ReceiverHypothesis,
    fec_hypotheses: List[Dict[str, Any]]
) -> List[ReceiverHypothesis]:
    """
    Hook for Stage 9 FEC candidate expansion.
    Leaves candidate intact and returns expansion stubs for downstream testing.
    """
    expanded = []
    for idx, fec_cfg in enumerate(fec_hypotheses):
        child = ReceiverHypothesis.from_dict(candidate.to_dict())
        child.candidate_id = f"{candidate.candidate_id}_fec{idx:02d}"
        child.status = CandidateStatus.FEC_TESTING.value
        child.fec_result = {"suggested_fec": fec_cfg}
        child.add_history_entry("fec_expansion", "EXPANDED", fec_cfg)
        expanded.append(child)
    return expanded
