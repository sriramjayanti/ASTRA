"""
Unit tests for candidate comparison, ranking margins, and why losers lost.
Tests:
- TEST 14: Candidate margin calculation
- TEST 15: Candidate tie handling
- TEST 17: Alternative hypotheses preservation
- TEST 20: Candidate comparison structure
"""

import pytest
from astra_explainability.src.candidate_comparison import CandidateComparator
from astra_explainability.src.utils import create_synthetic_perfect_record, create_tied_candidates_record


def test_20_candidate_comparison_structure():
    comparator = CandidateComparator()
    record = create_synthetic_perfect_record()
    candidates = record["stage_11_ranking"]["candidates"]

    compared = comparator.compare_candidates(candidates, top_k=3)
    assert len(compared) == 3

    # Best candidate
    c1 = compared[0]
    assert c1.is_best is True
    assert c1.rank == 1
    assert c1.score == 0.965
    assert c1.margin_to_next > 0.50
    assert c1.why_lost is None

    # Runner up (lost candidate)
    c2 = compared[1]
    assert c2.is_best is False
    assert c2.rank == 2
    assert c2.why_lost is not None
    assert "Ranked below winner" in c2.why_lost or "lower pipeline score" in c2.why_lost.lower()


def test_15_tie_handling():
    comparator = CandidateComparator()
    record = create_tied_candidates_record()
    candidates = record["stage_11_ranking"]["candidates"]

    compared = comparator.compare_candidates(candidates, top_k=3)
    c1 = compared[0]
    c2 = compared[1]

    # Margin is extremely small (0.005)
    assert c1.margin_to_next < 0.01
    assert "Virtually tied" in c2.why_lost or "close margin" in c2.why_lost.lower()


def test_entropy_and_alternatives():
    comparator = CandidateComparator()
    # High certainty: one dominant score
    ent_low = comparator.calculate_ranking_entropy([0.95, 0.03, 0.02])
    # High uncertainty: evenly distributed scores
    ent_high = comparator.calculate_ranking_entropy([0.34, 0.33, 0.33])

    assert ent_low < ent_high
