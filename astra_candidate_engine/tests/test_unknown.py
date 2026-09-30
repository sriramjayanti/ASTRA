"""
test_unknown.py
Tests for Unknown modulation and fallback rate handling.
"""

from astra_candidate_engine.src.inference import CandidateHypothesisEngine
from astra_candidate_engine.src.utils import create_mock_fusion_prediction, create_mock_symbol_rate_prediction


def test_unknown_modulation_route():
    engine = CandidateHypothesisEngine()
    fusion = create_mock_fusion_prediction([{"class": "UNKNOWN", "probability": 0.88}])
    rates = create_mock_symbol_rate_prediction([{"symbol_rate_hz": 9600, "score": 0.85}])
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    
    assert cand_set.candidates[0].modulation_family == "UNKNOWN"
    assert cand_set.candidates[0].requires_manual_or_custom_path is True


def test_fallback_rate_generation():
    engine = CandidateHypothesisEngine()
    fusion = create_mock_fusion_prediction([{"class": "QPSK", "probability": 0.80}])
    cand_set = engine.generate(fusion, symbol_rate_prediction=[], sample_rate_hz=192000.0)
    
    assert len(cand_set.candidates) > 0
    assert cand_set.generation_metadata["rate_source"] == "fallback_standard_grid"
