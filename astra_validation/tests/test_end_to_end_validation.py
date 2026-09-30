"""
test_end_to_end_validation.py
End-to-end multi-mechanism validation and comparative candidate ranking tests.
"""

import pytest
import numpy as np
from astra_validation.src.inference import ValidationEngine
from astra_validation.src.utils import generate_synthetic_stream
from astra_validation.src.models import ValidationStatus, EvidenceCheckState


def test_end_to_end_correct_candidate_validation():
    """Verify full synthetic stream achieves VALIDATION_STRONG with all independent checks passing."""
    engine = ValidationEngine()

    # Generate 8 synthetic telemetry frames (512 bits each = 4096 bits)
    stream = generate_synthetic_stream(
        num_frames=8,
        sync_pattern_hex="EB90",
        version=1,
        start_seq_id=10,
        payload_len=56,
        crc_profile_name="crc16_ccitt_false",
        seed=42,
    )

    mock_fec_candidate = {
        "candidate_id": "cand_correct",
        "demod_variant_id": "rot0",
        "interleaver_candidate_id": "block_16x16",
        "fec_candidate_id": "conv_k7_r12",
        "path_id": "QPSK_9600_rot0_block_conv",
        "fec_family": "convolutional",
        "fec_parameters": {"constraint_length": 7, "generators_octal": [0o171, 0o133]},
        "decoded_hard_bits": stream,
        "decoder_success": True,
        "path_metric": 0.02,
        "decoder_metrics": {"normalized_path_metric": 0.02, "termination_valid": True},
    }

    result = engine.validate(mock_fec_candidate, context={"candidate_frame_lengths": [512]})

    assert result.validation_status == ValidationStatus.VALIDATION_STRONG
    assert result.overall_validation_score >= 0.75
    assert result.strong_evidence_count >= 2
    assert result.contradiction_count == 0

    # Feature schema test
    feats = result.validation_features
    assert feats["crc_pass"] == 1.0
    assert feats["crc_pass_rate"] == 1.0
    assert feats["frame_periodicity_score"] > 0.10
    assert feats["overall_validation_score"] >= 0.75


def test_end_to_end_comparative_ranking_wrong_candidate():
    """Verify correct path outscores corrupted/wrong pipeline candidate paths."""
    engine = ValidationEngine()

    # True stream
    true_stream = generate_synthetic_stream(
        num_frames=8,
        sync_pattern_hex="EB90",
        version=1,
        start_seq_id=10,
        payload_len=56,
        crc_profile_name="crc16_ccitt_false",
        seed=42,
    )

    # Wrong candidate 1: inverted/phase rotated
    wrong_stream_1 = true_stream ^ 1

    # Wrong candidate 2: scrambled/wrong interleaver
    np.random.seed(99)
    perm = np.random.permutation(len(true_stream))
    wrong_stream_2 = true_stream[perm]

    cand_correct = {
        "candidate_id": "cand_correct",
        "path_id": "path_correct",
        "fec_family": "convolutional",
        "decoded_hard_bits": true_stream,
        "decoder_success": True,
        "path_metric": 0.02,
    }

    cand_wrong_1 = {
        "candidate_id": "cand_wrong_phase",
        "path_id": "path_wrong_phase",
        "fec_family": "convolutional",
        "decoded_hard_bits": wrong_stream_1,
        "decoder_success": False,
        "path_metric": 0.45,
    }

    cand_wrong_2 = {
        "candidate_id": "cand_wrong_interleaver",
        "path_id": "path_wrong_interleaver",
        "fec_family": "convolutional",
        "decoded_hard_bits": wrong_stream_2,
        "decoder_success": False,
        "path_metric": 0.50,
    }

    batch_results = engine.validate_batch([cand_correct, cand_wrong_1, cand_wrong_2])

    assert len(batch_results) == 3
    assert batch_results[0].candidate_id == "cand_correct"
    assert batch_results[0].overall_validation_score > batch_results[1].overall_validation_score
    assert batch_results[0].overall_validation_score > batch_results[2].overall_validation_score
