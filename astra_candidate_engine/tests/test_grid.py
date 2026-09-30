"""
test_grid.py
Tests for Cartesian product grid generation in Candidate / Hypothesis Engine.
"""

from astra_candidate_engine.src.inference import CandidateHypothesisEngine
from astra_candidate_engine.src.utils import create_mock_fusion_prediction, create_mock_symbol_rate_prediction


def test_grid_generation_dimensions():
    engine = CandidateHypothesisEngine()
    fusion = create_mock_fusion_prediction([
        {"class": "QPSK", "probability": 0.80},
        {"class": "8PSK", "probability": 0.12},
        {"class": "16-QAM", "probability": 0.05}
    ])
    rates = create_mock_symbol_rate_prediction([
        {"symbol_rate_hz": 9600, "score": 0.85},
        {"symbol_rate_hz": 4800, "score": 0.10},
        {"symbol_rate_hz": 19200, "score": 0.05}
    ])
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    assert cand_set.all_generated_count == 9
    assert len(cand_set.candidates) == 9
