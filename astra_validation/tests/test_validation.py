"""
test_validation.py
Master 30-Unit-Test Suite for ASTRA Stage 10 — Validation Engine.
Covers all 30 tests mandated in specification Section 98.
"""

import pytest
import numpy as np
import json

from astra_validation.src.models import (
    ValidationStatus,
    EvidenceCheckState,
    CRCProfile,
    CRCResult,
    ParityResult,
    SyndromeResult,
    ReencodingResult,
    FrameRepetitionResult,
    SyncWordResult,
    HeaderConsistencyResult,
    LengthConsistencyResult,
    StructuralResult,
    ValidationResult,
)
from astra_validation.src.crc import (
    load_crc_profiles,
    CRCCalculator,
    compute_crc,
    check_crc_frame,
    search_crc_candidates,
)
from astra_validation.src.parity import check_even_parity, check_odd_parity, check_block_parity
from astra_validation.src.syndrome import normalize_syndrome_evidence
from astra_validation.src.repetition import estimate_repetition_periods, score_frame_repetition
from astra_validation.src.sync_word import search_sync_word, score_sync_periodicity, hex_to_bits
from astra_validation.src.headers import validate_constant_field, validate_monotonic_counter
from astra_validation.src.length_checks import validate_length_field
from astra_validation.src.structure import extract_structural_evidence
from astra_validation.src.decoder_support import reencode_and_compare
from astra_validation.src.scoring import (
    evaluate_evidence_groups,
    compute_overall_validation_score,
    determine_validation_status,
    extract_stage11_features,
    VALIDATION_FEATURE_SCHEMA_VERSION,
)
from astra_validation.src.inference import ValidationEngine
from astra_validation.src.utils import generate_synthetic_frame, generate_synthetic_stream, compute_bitstream_hash


# TEST 1: known CRC vector
def test_1_known_crc_vector():
    profiles = load_crc_profiles()
    for name, prof in profiles.items():
        calc = CRCCalculator(prof)
        assert calc.verify_check_vector(), f"Failed on {name}"


# TEST 2: CRC positive frame
def test_2_crc_positive_frame():
    profiles = load_crc_profiles()
    prof = profiles["crc16_ccitt_false"]
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 10, dtype=np.uint8)
    crc_val = compute_crc(payload, prof)
    crc_bits = np.array([(crc_val >> i) & 1 for i in range(15, -1, -1)], dtype=np.uint8)
    frame = np.concatenate([payload, crc_bits])
    passed, _, _ = check_crc_frame(frame, prof, 16)
    assert passed is True


# TEST 3: CRC corrupted frame
def test_3_crc_corrupted_frame():
    profiles = load_crc_profiles()
    prof = profiles["crc16_ccitt_false"]
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 10, dtype=np.uint8)
    crc_val = compute_crc(payload, prof)
    crc_bits = np.array([(crc_val >> i) & 1 for i in range(15, -1, -1)], dtype=np.uint8)
    frame = np.concatenate([payload, crc_bits])
    frame[5] ^= 1 # flip bit
    passed, _, _ = check_crc_frame(frame, prof, 16)
    assert passed is False


# TEST 4: multiple CRC profiles
def test_4_multiple_crc_profiles():
    profiles = load_crc_profiles()
    assert "crc8_atm" in profiles
    assert "crc16_ccitt_false" in profiles
    assert "crc16_ibm" in profiles
    assert "crc32_ieee" in profiles


# TEST 5: CRC unavailable vs fail distinguished
def test_5_crc_unavailable_vs_fail():
    res_not_tested = CRCResult(profile_name="crc16", width=16, check_state=EvidenceCheckState.NOT_TESTED)
    res_fail = CRCResult(profile_name="crc16", width=16, checked_frames=4, passed_frames=0, check_state=EvidenceCheckState.FAIL)
    assert res_not_tested.check_state != res_fail.check_state


# TEST 6: even parity
def test_6_even_parity():
    # 8-bit blocks with even parity
    bits = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    res = check_even_parity(bits, block_size=8)
    assert res.pass_rate == 1.0
    assert res.check_state == EvidenceCheckState.PASS


# TEST 7: odd parity
def test_7_odd_parity():
    # 8-bit blocks with odd parity
    bits = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    res = check_odd_parity(bits, block_size=8)
    assert res.pass_rate == 1.0
    assert res.check_state == EvidenceCheckState.PASS


# TEST 8: block parity
def test_8_block_parity():
    mat = np.zeros((8, 8), dtype=np.uint8)
    res = check_block_parity(mat.ravel(), 8, 8)
    assert res.check_state == EvidenceCheckState.PASS


# TEST 9: syndrome normalization
def test_9_syndrome_normalization():
    mock = {"fec_family": "reed_solomon", "decoder_success": True, "syndrome_weight": 0.0}
    res = normalize_syndrome_evidence(mock)
    assert res.syndrome_valid is True
    assert res.normalized_syndrome_score == 1.0


# TEST 10: Viterbi metric normalization
def test_10_viterbi_metric_normalization():
    mock = {"fec_family": "convolutional", "decoder_success": True, "path_metric": 0.04}
    res = normalize_syndrome_evidence(mock)
    assert res.normalized_syndrome_score >= 0.8
    assert res.check_state == EvidenceCheckState.PASS


# TEST 11: re-encoding exact match
def test_11_reencoding_exact_match():
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    res = reencode_and_compare(msg, "none", {}, msg)
    assert res.bit_match_fraction == 1.0
    assert res.check_state == EvidenceCheckState.PASS


# TEST 12: re-encoding mismatch
def test_12_reencoding_mismatch():
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 8, dtype=np.uint8)
    corrupted = msg ^ 1
    res = reencode_and_compare(msg, "none", {}, corrupted)
    assert res.bit_match_fraction == 0.0
    assert res.check_state == EvidenceCheckState.FAIL


# TEST 13: frame periodicity detected
def test_13_frame_periodicity_detected():
    np.random.seed(42)
    hdr = np.array([1, 0, 1, 0] * 8, dtype=np.uint8)
    frames = [np.concatenate([hdr, np.random.randint(0, 2, 96, dtype=np.uint8)]) for _ in range(5)]
    stream = np.concatenate(frames)
    res = score_frame_repetition(stream, candidate_periods=[128])
    assert res.candidate_period_bits == 128
    assert res.check_state == EvidenceCheckState.PASS


# TEST 14: changing payload frame repetition
def test_14_changing_payload_frame_repetition():
    stream = generate_synthetic_stream(num_frames=6, payload_len=56, seed=10)
    res = score_frame_repetition(stream, candidate_periods=[512])
    assert res.repetition_count >= 5
    assert res.candidate_period_bits == 512


# TEST 15: sync word detection
def test_15_sync_word_detection():
    sync_bits = hex_to_bits("EB90")
    stream = np.concatenate([sync_bits, np.zeros(240, dtype=np.uint8), sync_bits, np.zeros(240, dtype=np.uint8)])
    res = search_sync_word(stream, "EB90")
    assert res.match_count == 2
    assert res.check_state == EvidenceCheckState.PASS


# TEST 16: sync tolerant Hamming detection
def test_16_sync_tolerant_hamming_detection():
    sync_bits = hex_to_bits("EB90").copy()
    sync_bits[0] ^= 1 # 1 bit error
    stream = np.concatenate([sync_bits, np.zeros(240, dtype=np.uint8)])
    res = search_sync_word(stream, "EB90", max_hamming_fraction=0.125)
    assert res.match_count == 1
    assert res.mean_hamming_distance == 1.0


# TEST 17: sync spacing consistency
def test_17_sync_spacing_consistency():
    positions = [0, 512, 1024, 1536]
    score = score_sync_periodicity(positions, 2048)
    assert score >= 0.90


# TEST 18: header constant-field check
def test_18_header_constant_field_check():
    frames = [np.array([0, 0, 0, 1] + [0] * 60, dtype=np.uint8) for _ in range(4)]
    res = validate_constant_field(frames, "version", 0, 4, expected_value=1)
    assert res.consistency_score == 1.0
    assert res.check_state == EvidenceCheckState.PASS


# TEST 19: header sequence check
def test_19_header_sequence_check():
    frames = []
    for i in range(4):
        f = np.zeros(64, dtype=np.uint8)
        f[7] = i & 1
        f[6] = (i >> 1) & 1
        frames.append(f)
    res = validate_monotonic_counter(frames, "seq", 0, 8)
    assert res.consistency_score == 1.0
    assert res.check_state == EvidenceCheckState.PASS


# TEST 20: length consistency
def test_20_length_consistency():
    frames = [np.zeros(512, dtype=np.uint8) for _ in range(4)]
    # Set declared length to 56 bytes at offset 32, length 8
    for f in frames:
        for b in range(8):
            f[32 + b] = (56 >> (7 - b)) & 1
    res = validate_length_field(frames, 32, 8, unit="bytes", observed_frame_length_bits=512)
    assert res.is_valid is True
    assert res.check_state == EvidenceCheckState.PASS


# TEST 21: random data does not crash
def test_21_random_data_does_not_crash():
    engine = ValidationEngine()
    np.random.seed(99)
    random_bits = np.random.randint(0, 2, size=1024, dtype=np.uint8)
    res = engine.validate(random_bits)
    assert res is not None
    assert res.validation_status in (ValidationStatus.VALIDATION_INCONCLUSIVE, ValidationStatus.VALIDATION_WEAK, ValidationStatus.VALIDATION_FAILED)


# TEST 22: inconclusive status
def test_22_inconclusive_status():
    status = determine_validation_status(overall_score=0.15, strong_evidence_groups=0, independent_evidence_groups=1, contradiction_count=0)
    assert status == ValidationStatus.VALIDATION_INCONCLUSIVE


# TEST 23: strong validation status
def test_23_strong_validation_status():
    status = determine_validation_status(overall_score=0.85, strong_evidence_groups=3, independent_evidence_groups=4, contradiction_count=0)
    assert status == ValidationStatus.VALIDATION_STRONG


# TEST 24: contradiction handling
def test_24_contradiction_handling():
    status = determine_validation_status(overall_score=0.10, strong_evidence_groups=0, independent_evidence_groups=1, contradiction_count=3)
    assert status == ValidationStatus.VALIDATION_FAILED


# TEST 25: independent evidence grouping
def test_25_independent_evidence_grouping():
    raw_c, indep_c, strong_c, contra, scores = evaluate_evidence_groups(
        crc_results=[CRCResult(profile_name="crc16", width=16, checked_frames=4, passed_frames=4, pass_rate=1.0, check_state=EvidenceCheckState.PASS)],
        parity_results=[],
        syndrome_result=SyndromeResult(syndrome_available=True, syndrome_valid=True, normalized_syndrome_score=1.0, check_state=EvidenceCheckState.PASS),
        reencoding_result=ReencodingResult(),
        repetition_result=FrameRepetitionResult(),
        sync_results=[],
        header_results=[],
        length_result=LengthConsistencyResult(),
        structural_result=StructuralResult(),
    )
    assert indep_c >= 2
    assert strong_c >= 2


# TEST 26: decoded-bit deduplication
def test_26_decoded_bit_deduplication():
    bits1 = np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8)
    bits2 = np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8)
    h1 = compute_bitstream_hash(bits1)
    h2 = compute_bitstream_hash(bits2)
    assert h1 == h2


# TEST 27: fixed feature schema
def test_27_fixed_feature_schema():
    engine = ValidationEngine()
    bits = np.array([1, 0] * 64, dtype=np.uint8)
    res = engine.validate(bits)
    feats = res.validation_features
    assert "crc_available" in feats
    assert "crc_pass" in feats
    assert "overall_validation_score" in feats
    assert "strong_evidence_groups" in feats


# TEST 28: Stage 9 integration
def test_28_stage_9_integration():
    engine = ValidationEngine()
    stream = generate_synthetic_stream(num_frames=4, payload_len=24)
    cand = {
        "candidate_id": "stage9_cand",
        "path_id": "p1",
        "fec_family": "reed_solomon",
        "decoded_hard_bits": stream,
        "decoder_success": True,
        "syndrome_weight": 0.0,
    }
    res = engine.validate(cand)
    assert res.syndrome_result.syndrome_valid is True


# TEST 29: batch validation
def test_29_batch_validation():
    engine = ValidationEngine()
    stream = generate_synthetic_stream(num_frames=4, payload_len=24)
    c1 = {"candidate_id": "c1", "path_id": "p1", "decoded_hard_bits": stream}
    c2 = {"candidate_id": "c2", "path_id": "p2", "decoded_hard_bits": stream ^ 1}
    results = engine.validate_batch([c1, c2])
    assert len(results) == 2
    assert results[0].candidate_id == "c1"


# TEST 30: JSON metadata serialization
def test_30_json_metadata_serialization():
    engine = ValidationEngine()
    stream = generate_synthetic_stream(num_frames=2, payload_len=24)
    res = engine.validate(stream)
    d = res.to_dict()
    json_str = json.dumps(d)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert "overall_validation_score" in parsed
