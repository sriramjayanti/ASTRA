"""
pruning.py
Duplicate rate merging, physical feasibility filtering, and beam-width pruning.
"""

from typing import List, Dict, Any, Tuple
from .models import ReceiverHypothesis, CandidateStatus
from .validators import validate_physical_feasibility


def merge_duplicate_rates(
    rate_candidates: List[Dict[str, Any]],
    tolerance_percent: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Merge symbol rate estimates that fall within a specified percentage tolerance.
    Example: 9580 Hz and 9600 Hz (0.2% difference) -> merged into highest-scoring candidate.
    """
    if not rate_candidates:
        return []

    # Sort descending by score first
    sorted_rates = sorted(rate_candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    merged: List[Dict[str, Any]] = []

    for rate_item in sorted_rates:
        r_hz = rate_item.get("symbol_rate_hz", 0.0)
        if r_hz <= 0:
            continue

        # Check if already close to an existing merged rate
        is_duplicate = False
        for existing in merged:
            e_hz = existing.get("symbol_rate_hz", 0.0)
            diff_pct = abs(r_hz - e_hz) / max(e_hz, 1e-6) * 100.0
            if diff_pct <= tolerance_percent:
                is_duplicate = True
                # Existing candidate already has higher or equal score because sorted
                break

        if not is_duplicate:
            merged.append(rate_item)

    return merged


def prune_and_rank_candidates(
    candidates: List[ReceiverHypothesis],
    beam_width: int = 5,
    max_candidates: int = 9,
    min_sps: float = 1.0,
    max_sps: float = 2000.0
) -> Tuple[List[ReceiverHypothesis], List[ReceiverHypothesis], List[ReceiverHypothesis]]:
    """
    1. Check physical validity of each candidate (hard invalidity check).
    2. Sort valid candidates descending by initial_score.
    3. Assign 1-indexed ranks.
    4. Apply soft beam-width pruning (mark pruned_by_beam=True beyond beam_width).
    5. Enforce max_candidates bound.

    Returns:
        (all_candidates, beam_surviving_candidates, rejected_candidates)
    """
    valid_list: List[ReceiverHypothesis] = []
    rejected_list: List[ReceiverHypothesis] = []

    for cand in candidates:
        is_valid, reason, sps = validate_physical_feasibility(
            symbol_rate_hz=cand.symbol_rate_hz,
            sample_rate_hz=cand.sample_rate_hz,
            min_sps=min_sps,
            max_sps=max_sps
        )
        cand.samples_per_symbol = sps

        if not is_valid:
            cand.rejected = True
            cand.rejection_reason = reason
            cand.status = CandidateStatus.REJECTED.value
            cand.add_history_entry(stage="physical_validation", status="REJECTED", details={"reason": reason})
            rejected_list.append(cand)
        else:
            cand.rejected = False
            cand.rejection_reason = None
            valid_list.append(cand)

    # Sort valid candidates descending by score
    valid_list.sort(key=lambda x: x.initial_score, reverse=True)

    # Hard cap on candidate count
    if len(valid_list) > max_candidates:
        capped_out = valid_list[max_candidates:]
        for c in capped_out:
            c.pruned_by_beam = True
            c.add_history_entry(stage="max_candidates_cap", status="PRUNED")
        valid_list = valid_list[:max_candidates]

    # Assign ranks and apply beam pruning
    beam_survivors: List[ReceiverHypothesis] = []
    for rank_idx, cand in enumerate(valid_list, start=1):
        cand.candidate_rank = rank_idx
        if rank_idx <= beam_width:
            cand.pruned_by_beam = False
            cand.add_history_entry(stage="beam_pruning", status="KEPT_IN_BEAM", details={"rank": rank_idx})
            beam_survivors.append(cand)
        else:
            cand.pruned_by_beam = True
            cand.add_history_entry(stage="beam_pruning", status="PRUNED_BY_BEAM", details={"rank": rank_idx})

    all_processed = valid_list + rejected_list
    return all_processed, beam_survivors, rejected_list
