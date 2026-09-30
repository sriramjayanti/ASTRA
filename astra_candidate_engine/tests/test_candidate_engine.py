"""
test_candidate_engine.py
Master test suite for ASTRA Stage 5 Candidate / Hypothesis Engine covering all 20 mandatory tests.
Works with both pytest and standard unittest.
"""

import math
import json
from astra_candidate_engine.src.models import ReceiverHypothesis, CandidateSet, CandidateStatus
from astra_candidate_engine.src.inference import CandidateHypothesisEngine
from astra_candidate_engine.src.scoring import calculate_initial_score
from astra_candidate_engine.src.pruning import merge_duplicate_rates
from astra_candidate_engine.src.mappings import get_demod_hints, get_sync_hints, get_modulation_family
from astra_candidate_engine.src.utils import (
    create_mock_fusion_prediction,
    create_mock_symbol_rate_prediction,
    create_mock_rf_support,
    create_mock_constellation_evidence,
)


def get_default_engine():
    return CandidateHypothesisEngine()


# TEST 1: 3x3 grid creates 9 candidates
def test_1_3x3_grid_creates_9_candidates(engine=None):
    engine = engine or get_default_engine()
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


# TEST 2: correct modulation/rate pairing
def test_2_correct_modulation_rate_pairing(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction([{"class": "QPSK", "probability": 0.80}])
    rates = create_mock_symbol_rate_prediction([{"symbol_rate_hz": 9600, "score": 0.85}])
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    assert len(cand_set.candidates) == 1
    c = cand_set.candidates[0]
    assert c.modulation == "QPSK"
    assert c.symbol_rate_hz == 9600.0
    assert c.modulation_probability == 0.80
    assert c.symbol_rate_score == 0.85


# TEST 3: initial score calculation
def test_3_initial_score_calculation():
    # Base product: 0.80 * 0.85 = 0.68
    score = calculate_initial_score(0.80, 0.85)
    assert abs(score - 0.68) < 1e-4

    # With RF and Constellation support
    supported_score = calculate_initial_score(
        0.80, 0.85, rf_family_prob=0.92, constellation_score=0.88,
        config={"score": {"modulation_weight": 1.0, "symbol_rate_weight": 1.0, "rf_family_weight": 0.25, "constellation_weight": 0.25}}
    )
    assert 0.0 < supported_score <= 1.0
    assert supported_score > 0.60


# TEST 4: ranking descending
def test_4_ranking_descending(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    
    scores = [c.initial_score for c in cand_set.candidates if not c.rejected]
    assert scores == sorted(scores, reverse=True)
    assert cand_set.candidates[0].candidate_rank == 1


# TEST 5: beam pruning
def test_5_beam_pruning(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    
    # beam_width = 5 by default
    assert len(cand_set.beam_candidates) == 5
    for c in cand_set.beam_candidates:
        assert not c.pruned_by_beam
    for c in cand_set.candidates[5:]:
        assert c.pruned_by_beam


# TEST 6: duplicate rates merged
def test_6_duplicate_rates_merged():
    rates = [
        {"symbol_rate_hz": 9600.0, "score": 0.85},
        {"symbol_rate_hz": 9580.0, "score": 0.50}, # 0.2% diff -> duplicate
        {"symbol_rate_hz": 4800.0, "score": 0.20},
    ]
    merged = merge_duplicate_rates(rates, tolerance_percent=2.0)
    assert len(merged) == 2
    merged_rates = [r["symbol_rate_hz"] for r in merged]
    assert 9600.0 in merged_rates
    assert 4800.0 in merged_rates


# TEST 7: invalid rates rejected
def test_7_invalid_rates_rejected(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction([{"class": "QPSK", "probability": 0.80}])
    rates = [
        {"symbol_rate_hz": -9600.0, "score": 0.85}, # Negative rate
        {"symbol_rate_hz": 300000.0, "score": 0.50}, # Fs = 192k -> SPS = 0.64 (< 1.0)
    ]
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    assert cand_set.rejected_candidate_count >= 1
    for r in cand_set.candidates:
        if r.rejected:
            assert r.rejection_reason is not None
            assert r.status == CandidateStatus.REJECTED.value


# TEST 8: non-integer SPS
def test_8_non_integer_sps(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction([{"class": "QPSK", "probability": 0.80}])
    rates = [{"symbol_rate_hz": 7000.0, "score": 0.80}] # 192000 / 7000 = 27.42857...
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    c = cand_set.candidates[0]
    assert abs(c.samples_per_symbol - (192000.0 / 7000.0)) < 1e-4
    assert not float(c.samples_per_symbol).is_integer()


# TEST 9: RF evidence optional
def test_9_rf_evidence_optional(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    # Without RF
    set_without_rf = engine.generate(fusion, rates, sample_rate_hz=192000.0, rf_support=None)
    assert set_without_rf.candidates[0].rf_family_probability is None

    # With RF
    rf_sup = create_mock_rf_support({"PSK": 0.95})
    set_with_rf = engine.generate(fusion, rates, sample_rate_hz=192000.0, rf_support=rf_sup)
    assert set_with_rf.candidates[0].rf_family_probability == 0.95


# TEST 10: constellation evidence optional
def test_10_constellation_evidence_optional(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    # Without constellation
    set_no_const = engine.generate(fusion, rates, sample_rate_hz=192000.0, constellation_evidence=None)
    assert set_no_const.candidates[0].constellation_support_score is None

    # With constellation
    const_ev = create_mock_constellation_evidence({"QPSK": 0.88})
    set_const = engine.generate(fusion, rates, sample_rate_hz=192000.0, constellation_evidence=const_ev)
    assert set_const.candidates[0].constellation_support_score == 0.88


# TEST 11: Unknown modulation handling
def test_11_unknown_modulation_handling(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction([{"class": "UNKNOWN", "probability": 0.90}])
    rates = create_mock_symbol_rate_prediction([{"symbol_rate_hz": 9600, "score": 0.85}])
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    c = cand_set.candidates[0]
    assert c.modulation_family == "UNKNOWN"
    assert not c.demodulator_supported
    assert c.requires_manual_or_custom_path


# TEST 12: fallback rate mode
def test_12_fallback_rate_mode(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction([{"class": "QPSK", "probability": 0.80}])
    # Pass empty rates
    cand_set = engine.generate(fusion, symbol_rate_prediction=[], sample_rate_hz=192000.0)
    assert len(cand_set.candidates) > 0
    assert cand_set.generation_metadata["rate_source"] == "fallback_standard_grid"
    assert cand_set.candidates[0].source_evidence["symbol_rate_branch"]["source"] == "fallback_standard_grid"


# TEST 13: source evidence preserved
def test_13_source_evidence_preserved(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    rf = create_mock_rf_support()
    const = create_mock_constellation_evidence()
    
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0, rf_support=rf, constellation_evidence=const)
    c = cand_set.candidates[0]
    assert "modulation_branch" in c.source_evidence
    assert "symbol_rate_branch" in c.source_evidence
    assert "rf_family_support" in c.source_evidence
    assert "constellation_support" in c.source_evidence


# TEST 14: candidate IDs unique
def test_14_candidate_ids_unique(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    
    ids = [c.candidate_id for c in cand_set.candidates]
    assert len(ids) == len(set(ids))


# TEST 15: JSON serialization
def test_15_json_serialization(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    
    json_str = cand_set.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert "candidates" in parsed
    assert len(parsed["candidates"]) == len(cand_set.candidates)


# TEST 16: deterministic output
def test_16_deterministic_output(engine=None):
    engine = engine or get_default_engine()
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    
    set1 = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    set2 = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    
    assert [c.candidate_id for c in set1.candidates] == [c.candidate_id for c in set2.candidates]
    assert [c.initial_score for c in set1.candidates] == [c.initial_score for c in set2.candidates]


# TEST 17: max candidate bound respected
def test_17_max_candidate_bound_respected():
    # Configure max_candidates = 4
    engine_capped = CandidateHypothesisEngine(config={
        "candidate_engine": {"max_candidates": 4, "beam_width": 2, "modulation_top_k": 3, "symbol_rate_top_k": 3}
    })
    fusion = create_mock_fusion_prediction()
    rates = create_mock_symbol_rate_prediction()
    cand_set = engine_capped.generate(fusion, rates, sample_rate_hz=192000.0)
    
    assert len(cand_set.candidates) == 4
    assert len(cand_set.beam_candidates) == 2


# TEST 18: rate variants controlled
def test_18_rate_variants_controlled():
    # Enable rate variants [-1%, 0%, +1%]
    engine_variants = CandidateHypothesisEngine(config={
        "candidate_engine": {
            "rate_variants": {
                "enabled": True,
                "offsets_percent": [-1.0, 0.0, 1.0]
            },
            "max_candidates": 27,
            "beam_width": 10
        }
    })
    fusion = create_mock_fusion_prediction([{"class": "QPSK", "probability": 0.80}])
    rates = create_mock_symbol_rate_prediction([{"symbol_rate_hz": 9600.0, "score": 0.85}])
    cand_set = engine_variants.generate(fusion, rates, sample_rate_hz=192000.0)
    
    # 1 mod * 1 rate * 3 variants = 3 candidates
    assert cand_set.all_generated_count == 3
    rates_gen = sorted([round(c.symbol_rate_hz) for c in cand_set.candidates])
    assert rates_gen == [9504, 9600, 9696]


# TEST 19: unsupported modulation routing
def test_19_unsupported_modulation_routing(engine=None):
    # Analog modulation (e.g. WBFM)
    hints = get_demod_hints("WBFM")
    assert not hints["supported"]
    assert hints["demodulator_type"] == "analog_demodulator"

    # FSK routing
    fsk_sync = get_sync_hints("2-FSK")
    assert fsk_sync["family"] == "FSK"
    assert "early_late" in fsk_sync["timing_recovery_methods"]

    # PSK routing
    psk_sync = get_sync_hints("QPSK")
    assert psk_sync["family"] == "PSK"
    assert psk_sync["costas_order"] == 4


# TEST 20: no NaN/Inf scores
def test_20_no_nan_inf_scores(engine=None):
    engine = engine or get_default_engine()
    fusion = {"top_k": [{"class": "QPSK", "probability": float('nan')}]}
    rates = {"top_k": [{"symbol_rate_hz": float('inf'), "score": float('nan')}]}
    
    cand_set = engine.generate(fusion, rates, sample_rate_hz=192000.0)
    for c in cand_set.candidates:
        assert not math.isnan(c.initial_score)
        assert not math.isinf(c.initial_score)
        assert 0.0 <= c.initial_score <= 1.0
