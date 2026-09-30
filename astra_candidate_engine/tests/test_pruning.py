"""
test_pruning.py
Tests for physical validation and beam-width pruning.
"""

from astra_candidate_engine.src.inference import CandidateHypothesisEngine
from astra_candidate_engine.src.utils import create_mock_fusion_prediction, create_mock_symbol_rate_prediction


def test_beam_width_pruning():
    engine = CandidateHypothesisEngine(config={
        "candidate_engine": {"beam_width": 3, "max_candidates": 9}
    })
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    
    assert len(cand_set.beam_candidates) == 3
    assert len(cand_set.candidates) == 9
    assert cand_set.pruned_candidate_count == 6
