"""
ASTRA Stage 5 — Candidate / Hypothesis Engine
"""

from .models import ReceiverHypothesis, CandidateSet, CandidateStatus
from .inference import CandidateHypothesisEngine
from .mappings import get_modulation_family, get_demod_hints, get_sync_hints
from .scoring import calculate_initial_score
from .pruning import merge_duplicate_rates, prune_and_rank_candidates
from .explainability import explain_candidate, format_candidate_summary_table
from .lifecycle import (
    update_candidate_status,
    record_sync_result,
    record_demod_result,
    record_validation_result,
    expand_with_interleavers,
    expand_with_fec,
)

__all__ = [
    "CandidateHypothesisEngine",
    "ReceiverHypothesis",
    "CandidateSet",
    "CandidateStatus",
    "get_modulation_family",
    "get_demod_hints",
    "get_sync_hints",
    "calculate_initial_score",
    "merge_duplicate_rates",
    "prune_and_rank_candidates",
    "explain_candidate",
    "format_candidate_summary_table",
    "update_candidate_status",
    "record_sync_result",
    "record_demod_result",
    "record_validation_result",
    "expand_with_interleavers",
    "expand_with_fec",
]
