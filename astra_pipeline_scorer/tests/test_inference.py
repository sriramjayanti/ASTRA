"""
test_inference.py
Unit tests for PipelineScoringEngine and rule-based fallback.
"""

import pytest
from astra_pipeline_scorer.src.inference import PipelineScoringEngine
from astra_pipeline_scorer.src.utils import generate_synthetic_candidate_tree


def test_rule_based_fallback_when_no_model():
    engine = PipelineScoringEngine() # No checkpoint
    cands, _ = generate_synthetic_candidate_tree("sig_fallback", num_competing_candidates=5)

    res = engine.rank(cands, signal_id="sig_fallback", top_k=3)
    assert res.fallback_used is True
    assert len(res.ranked_candidates) == 3
    assert res.ranked_candidates[0].candidate_id == "cand_correct"
