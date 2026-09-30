"""
test_scoring.py
Unit tests for multi-mechanism scoring, evidence grouping, and contradiction tracking.
"""

import pytest
from astra_validation.src.models import (
    ValidationStatus,
    EvidenceCheckState,
    CRCResult,
    ParityResult,
    SyndromeResult,
    ReencodingResult,
    FrameRepetitionResult,
    SyncWordResult,
    HeaderConsistencyResult,
    LengthConsistencyResult,
    StructuralResult,
)
from astra_validation.src.scoring import (
    evaluate_evidence_groups,
    compute_overall_validation_score,
    determine_validation_status,
)


def test_evidence_grouping_and_strong_status():
    # Strong CRC + Valid Syndrome + Strong Sync Word
    crc_res = [CRCResult(profile_name="crc16_ccitt_false", width=16, checked_frames=8, passed_frames=8, pass_rate=1.0, check_state=EvidenceCheckState.PASS)]
    synd_res = SyndromeResult(syndrome_available=True, syndrome_valid=True, normalized_syndrome_score=1.0, check_state=EvidenceCheckState.PASS)
    sync_res = [SyncWordResult(pattern_id="sync_EB90", match_count=8, periodicity_score=0.95, check_state=EvidenceCheckState.PASS)]
    rep_res = FrameRepetitionResult(candidate_period_bits=512, repetition_count=8, periodicity_score=0.75, check_state=EvidenceCheckState.PASS)
    hdr_res = [HeaderConsistencyResult(field_name="version", consistency_score=1.0, check_state=EvidenceCheckState.PASS)]
    len_res = LengthConsistencyResult(declared_length=56, observed_length=64, is_valid=True, check_state=EvidenceCheckState.PASS)
    reenc_res = ReencodingResult(bit_match_fraction=0.99, check_state=EvidenceCheckState.PASS)
    struct_res = StructuralResult(binary_entropy=0.96)

    raw_c, indep_c, strong_c, contra, group_scores = evaluate_evidence_groups(
        crc_results=crc_res,
        parity_results=[],
        syndrome_result=synd_res,
        reencoding_result=reenc_res,
        repetition_result=rep_res,
        sync_results=sync_res,
        header_results=hdr_res,
        length_result=len_res,
        structural_result=struct_res,
    )

    assert strong_c >= 4
    assert len(contra) == 0

    score = compute_overall_validation_score(group_scores, len(contra))
    assert score >= 0.85

    status = determine_validation_status(score, strong_c, indep_c, len(contra))
    assert status == ValidationStatus.VALIDATION_STRONG


def test_inconclusive_status_on_blind_data():
    # No CRC, no sync, no repetition, but normal entropy
    raw_c, indep_c, strong_c, contra, group_scores = evaluate_evidence_groups(
        crc_results=[],
        parity_results=[],
        syndrome_result=SyndromeResult(check_state=EvidenceCheckState.NOT_TESTED),
        reencoding_result=ReencodingResult(check_state=EvidenceCheckState.NOT_TESTED),
        repetition_result=FrameRepetitionResult(check_state=EvidenceCheckState.NOT_TESTED),
        sync_results=[],
        header_results=[],
        length_result=LengthConsistencyResult(check_state=EvidenceCheckState.NOT_TESTED),
        structural_result=StructuralResult(binary_entropy=0.99),
    )

    assert strong_c == 0
    assert len(contra) == 0

    score = compute_overall_validation_score(group_scores, len(contra))
    status = determine_validation_status(score, strong_c, indep_c, len(contra))
    assert status == ValidationStatus.VALIDATION_INCONCLUSIVE


def test_contradiction_handling_and_failed_status():
    # CRC failed + Length impossible
    crc_res = [CRCResult(profile_name="crc16_ccitt_false", width=16, checked_frames=6, passed_frames=0, pass_rate=0.0, check_state=EvidenceCheckState.FAIL)]
    len_res = LengthConsistencyResult(declared_length=200, observed_length=64, is_valid=False, check_state=EvidenceCheckState.FAIL)

    raw_c, indep_c, strong_c, contra, group_scores = evaluate_evidence_groups(
        crc_results=crc_res,
        parity_results=[],
        syndrome_result=SyndromeResult(check_state=EvidenceCheckState.FAIL),
        reencoding_result=ReencodingResult(bit_match_fraction=0.45, check_state=EvidenceCheckState.FAIL),
        repetition_result=FrameRepetitionResult(check_state=EvidenceCheckState.NOT_TESTED),
        sync_results=[],
        header_results=[],
        length_result=len_res,
        structural_result=StructuralResult(binary_entropy=0.50),
    )

    assert len(contra) >= 2
    score = compute_overall_validation_score(group_scores, len(contra))
    status = determine_validation_status(score, strong_c, indep_c, len(contra))
    assert status == ValidationStatus.VALIDATION_FAILED
