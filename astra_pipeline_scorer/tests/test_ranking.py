"""
test_ranking.py
Unit tests for candidate ranking, sorting, score margin, and Top-K selection.
"""

import pytest
import numpy as np
from astra_pipeline_scorer.src.ranking import rank_pipeline_candidates, compute_ranking_uncertainty, determine_confidence_tier
from astra_pipeline_scorer.src.models import ConfidenceTier


def test_candidate_ranking_descending_and_top_k():
    candidates = [
        {"candidate_id": "c1", "modulation": "QPSK"},
        {"candidate_id": "c2", "modulation": "8PSK"},
        {"candidate_id": "c3", "modulation": "16QAM"},
        {"candidate_id": "c4", "modulation": "2FSK"},
    ]
    scores = np.array([0.90, 0.20, 0.70, 0.10])

    res = rank_pipeline_candidates(candidates, scores, top_k=3, signal_id="sig_test")
    assert res.candidate_count == 4
    assert len(res.ranked_candidates) == 3
    assert res.ranked_candidates[0].candidate_id == "c1"
    assert res.ranked_candidates[1].candidate_id == "c3"
    assert res.ranked_candidates[2].candidate_id == "c2"
    assert res.top1_score == 0.90
    assert res.top2_score == 0.70
    assert abs(res.score_margin - 0.20) < 1e-4


def test_score_margin_and_uncertainty():
    tied_scores = [0.45, 0.44, 0.43]
    dom_scores = [0.95, 0.05, 0.01]

    u_tied = compute_ranking_uncertainty(tied_scores)
    u_dom = compute_ranking_uncertainty(dom_scores)

    assert u_tied > u_dom
