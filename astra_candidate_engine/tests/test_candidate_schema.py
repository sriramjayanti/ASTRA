"""
test_candidate_schema.py
Tests for ReceiverHypothesis and CandidateSet schema serialization and lifecycle tracking.
"""

from astra_candidate_engine.src.models import ReceiverHypothesis, CandidateSet, CandidateStatus
from astra_candidate_engine.src.lifecycle import (
    record_sync_result,
    record_demod_result,
    expand_with_interleavers,
    expand_with_fec,
)


def test_schema_dict_and_json_roundtrip():
    cand = ReceiverHypothesis(
        candidate_id="cand_QPSK_9600_001",
        modulation="QPSK",
        modulation_probability=0.80,
        symbol_rate_hz=9600.0,
        symbol_rate_score=0.85,
        samples_per_symbol=20.0,
        sample_rate_hz=192000.0,
        modulation_family="PSK",
        initial_score=0.68
    )
    d = cand.to_dict()
    assert d["candidate_id"] == "cand_QPSK_9600_001"
    assert d["initial_score"] == 0.68
    
    cand_set = CandidateSet(candidates=[cand], beam_candidates=[cand], all_generated_count=1)
    json_repr = cand_set.to_json()
    assert "cand_QPSK_9600_001" in json_repr


def test_candidate_lifecycle_and_expansion():
    cand = ReceiverHypothesis(
        candidate_id="cand_QPSK_9600_001",
        modulation="QPSK",
        symbol_rate_hz=9600.0
    )
    # Record sync success
    cand = record_sync_result(cand, passed=True, sync_metrics={"cfo_hz": 12.5, "timing_lock": 0.94})
    assert cand.status == CandidateStatus.SYNC_PASSED.value
    assert cand.sync_result["timing_lock"] == 0.94

    # Record demod success
    cand = record_demod_result(cand, passed=True, demod_metrics={"evm_db": -24.5})
    assert cand.status == CandidateStatus.DEMOD_PASSED.value

    # Expand interleavers
    il_children = expand_with_interleavers(cand, [{"type": "block", "depth": 8}])
    assert len(il_children) == 1
    assert il_children[0].status == CandidateStatus.INTERLEAVER_TESTING.value
