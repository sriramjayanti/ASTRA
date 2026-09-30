"""
test_end_to_end_bitstream.py
Comprehensive end-to-end unit and integration tests for ASTRA Stage 12 Bitstream Intelligence Engine.
"""

import pytest
import numpy as np
import json

from astra_bitstream_intelligence.src.models import StructureStatus, BitstreamIntelligenceResult
from astra_bitstream_intelligence.src.inference import BitstreamIntelligenceEngine
from astra_bitstream_intelligence.src.bit_balance import calculate_bit_balance
from astra_bitstream_intelligence.src.run_length import calculate_run_length_stats
from astra_bitstream_intelligence.src.segmentation import (
    build_frame_matrix,
    compute_position_stability,
    compute_position_entropy,
    compute_xor_frame_differences
)
from astra_bitstream_intelligence.src.utils import generate_synthetic_framed_bitstream
from astra_bitstream_intelligence.src.feature_builder import FEATURE_SCHEMA_VERSION


def test_17_run_length_statistics():
    """TEST 17: run-length statistics calculation."""
    # 0000 1111 00 1 00000000 (8 zeros)
    bits = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    stats = calculate_run_length_stats(bits, long_run_threshold=8)
    assert stats.max_run_length == 8
    assert stats.long_run_fraction > 0.0
    assert stats.mean_run_length > 1.0


def test_18_bit_balance():
    """TEST 18: bit balance calculation."""
    bits = np.array([1, 1, 1, 0], dtype=np.uint8)
    bal = calculate_bit_balance(bits)
    assert bal.one_fraction == 0.75
    assert bal.zero_fraction == 0.25
    assert bal.imbalance == 0.25


def test_20_position_entropy_map():
    """TEST 20: position entropy map across candidate frames."""
    # 4 frames of 4 bits: bit 0 is fixed 1, bit 1 is variable
    mat = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 1, 0]
    ], dtype=np.uint8)

    ent = compute_position_entropy(mat)
    assert ent[0] == 0.0 # Fixed bit has 0 entropy
    assert ent[2] == 0.0 # Fixed bit has 0 entropy
    assert ent[1] == 1.0 # 50/50 bit has max entropy 1.0


def test_21_frame_stability_map():
    """TEST 21: frame stability map calculation."""
    mat = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 1, 0]
    ], dtype=np.uint8)

    stab = compute_position_stability(mat)
    assert stab[0] == 1.0 # Fixed 1 -> stability 1.0
    assert stab[1] == 0.0 # 50% 1 -> stability 0.0


def test_22_xor_frame_comparison():
    """TEST 22: XOR frame comparison reveals unchanging vs changing positions."""
    mat = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ], dtype=np.uint8)

    xor_diff = compute_xor_frame_differences(mat)
    assert xor_diff.shape == (2, 4)
    # Positions 0 and 2 do not change between any consecutive frames -> XOR is all 0
    assert np.all(xor_diff[:, 0] == 0)
    assert np.all(xor_diff[:, 2] == 0)


def test_23_short_stream_partial_result():
    """TEST 23: short bitstream returns partial metrics without crash."""
    engine = BitstreamIntelligenceEngine()
    short_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    res = engine.analyze(short_bits)

    assert res.bit_count == 16
    assert res.binary_entropy_global > 0.0
    assert res.status in [StructureStatus.STRUCTURE_UNKNOWN, StructureStatus.STRUCTURE_WEAK]


def test_24_random_stream_returns_unknown_or_weak():
    """TEST 24: pure random bitstream returns STRUCTURE_UNKNOWN or WEAK."""
    engine = BitstreamIntelligenceEngine()
    rng = np.random.RandomState(42)
    rand_bits = rng.randint(0, 2, size=4096, dtype=np.uint8)

    res = engine.analyze(rand_bits)
    assert res.status in [StructureStatus.STRUCTURE_UNKNOWN, StructureStatus.STRUCTURE_WEAK]
    assert res.binary_entropy_global > 0.98


def test_25_invalid_bits_rejected():
    """TEST 25: NaN or non-binary bits are rejected cleanly."""
    engine = BitstreamIntelligenceEngine()
    invalid_bits = np.array([0, 1, 2, 3], dtype=np.uint8)
    with pytest.raises(ValueError):
        engine.analyze(invalid_bits)

    nan_bits = np.array([np.nan, 1.0, 0.0])
    with pytest.raises(ValueError):
        engine.analyze(nan_bits)


def test_27_stage_11_candidate_integration():
    """TEST 27: Stage 11 candidate object integration."""
    engine = BitstreamIntelligenceEngine()
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=20,
        sync_word_hex="EB90",
        seed=42
    )

    # Mock Stage 11 Candidate Pipeline Result object
    class MockStage11Candidate:
        def __init__(self, c_id, b):
            self.pipeline_path_id = c_id
            self.decoded_hard_bits = b
            self.soft_information = np.ones(len(b), dtype=np.float32)

    cand = MockStage11Candidate("pipe_opt_1", bits)
    res = engine.analyze_candidate(cand)

    assert res.pipeline_path_id == "pipe_opt_1"
    assert res.status == StructureStatus.STRUCTURE_STRONG
    assert len(res.frame_length_candidates) > 0
    assert res.frame_length_candidates[0].period_bits == 512


def test_28_deduplicated_bitstream_analysis_reuse():
    """TEST 28: identical decoded bitstreams reuse cached analysis."""
    engine = BitstreamIntelligenceEngine()
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=256,
        num_frames=10,
        sync_word_hex="EB90",
        seed=11
    )

    res1 = engine.analyze(bits, context={"pipeline_path_id": "cand_1"})
    res2 = engine.analyze(bits, context={"pipeline_path_id": "cand_2"})

    assert res1.decoded_bits_hash == res2.decoded_bits_hash
    assert res2.pipeline_path_id == "cand_2"


def test_29_feature_schema_stability():
    """TEST 29: feature schema stability for Stage 13."""
    engine = BitstreamIntelligenceEngine()
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=10,
        sync_word_hex="EB90",
        seed=77
    )
    res = engine.analyze(bits)

    assert FEATURE_SCHEMA_VERSION == "bitstream_features_v1"
    gf = res.global_features
    assert "binary_entropy_global" in gf
    assert "top_frame_length" in gf
    assert "periodicity_strength" in gf
    assert res.sequence_feature_map is not None
    assert res.sequence_feature_map.shape == (len(bits), 4)


def test_30_json_metadata_serializable():
    """TEST 30: Result metadata is fully JSON serializable."""
    engine = BitstreamIntelligenceEngine()
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=10,
        sync_word_hex="EB90",
        seed=88
    )
    res = engine.analyze(bits)

    res_dict = res.to_dict(include_sequence_maps=True)
    json_str = json.dumps(res_dict, indent=2)
    assert len(json_str) > 100
    loaded = json.loads(json_str)
    assert loaded["status"] == "STRUCTURE_STRONG"
