"""
test_scoring.py
Tests for multi-evidence score calculation.
"""

from astra_candidate_engine.src.scoring import calculate_initial_score


def test_weighted_product_scoring():
    score = calculate_initial_score(0.80, 0.85)
    assert abs(score - 0.68) < 1e-4


def test_scoring_with_support_evidence():
    score = calculate_initial_score(0.80, 0.85, rf_family_prob=0.95, constellation_score=0.90)
    assert score > 0.60
