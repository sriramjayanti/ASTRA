import pytest
import numpy as np
from astra_payload_explorer.src.models import ProtocolProfile, FieldDefinition, EvidenceLevel
from astra_payload_explorer.src.frame_resolver import resolve_frame_hypotheses, select_frame_segmentation
from astra_bitstream_intelligence.src.models import BitstreamIntelligenceResult, PeriodicityCandidate, SyncPatternResult
from astra_bitstream_transformer.src.models import BitstreamStructurePrediction, RegionPrediction

def test_resolve_from_known_profile():
    profile = ProtocolProfile(
        profile_id="astra_test",
        frame_length_bits=256,
        sync_length_bits=32,
        header_length_bits=32,
        sync_pattern="10101010101010101010101010101010"
    )
    # Stream with sync pattern at bit 16
    sync_bits = [1, 0] * 16
    bits = np.array([0]*16 + sync_bits + [0]*(256 - 32) + sync_bits + [0]*(256 - 32), dtype=np.uint8)

    candidates = resolve_frame_hypotheses(
        decoded_bits=bits,
        stage12_result=None,
        stage13_result=None,
        validation_result=None,
        profile=profile
    )
    assert len(candidates) > 0
    top_cand = candidates[0]
    assert top_cand.length_bits == 256
    assert top_cand.alignment_offset_bits == 16
    assert top_cand.support_score >= 0.8

def test_resolve_from_stage12_and_13():
    # Stage 12 indicates 128 bit frame periodicity
    stage12 = BitstreamIntelligenceResult(
        pipeline_path_id="test_pipe",
        bit_count=512,
        periodicity_candidates=[
            PeriodicityCandidate(period_bits=128, score=0.9, support_sources=["autocorr"])
        ],
        sync_results=[
            SyncPatternResult(pattern_id="sync_cand_1", positions=[0], confidence=0.85)
        ]
    )
    # 512 bits
    bits = np.random.randint(0, 2, size=512, dtype=np.uint8)

    candidates = resolve_frame_hypotheses(
        decoded_bits=bits,
        stage12_result=stage12,
        stage13_result=None,
        validation_result=None,
        profile=None
    )
    assert len(candidates) > 0
    assert any(c.length_bits == 128 for c in candidates)
    best = select_frame_segmentation(candidates)
    assert best is not None
    assert best.length_bits == 128

def test_boundary_disagreement_detection():
    # Stage 13 predicts frame length 200 vs Stage 12 predicts 256
    stage12 = BitstreamIntelligenceResult(
        pipeline_path_id="test_pipe_disagree",
        bit_count=600,
        periodicity_candidates=[
            PeriodicityCandidate(period_bits=256, score=0.8, support_sources=["autocorr"])
        ]
    )
    # Stage 13 predicts sync repeats
    stage13 = BitstreamStructurePrediction(
        pipeline_path_id="test_pipe_disagree",
        sequence_length=600,
        predicted_labels=np.zeros(600, dtype=np.int64),
        label_probabilities=np.zeros((600, 6), dtype=np.float32),
        regions=[
            RegionPrediction(start_bit=0, end_bit=31, label="SYNC", label_id=1, mean_probability=0.95, min_probability=0.9),
            RegionPrediction(start_bit=200, end_bit=231, label="SYNC", label_id=1, mean_probability=0.95, min_probability=0.9),
        ]
    )
    bits = np.zeros(600, dtype=np.uint8)
    candidates = resolve_frame_hypotheses(
        decoded_bits=bits,
        stage12_result=stage12,
        stage13_result=stage13,
        validation_result=None,
        profile=None
    )
    lengths = [c.length_bits for c in candidates]
    assert 200 in lengths or 256 in lengths
    assert len(candidates) >= 1
