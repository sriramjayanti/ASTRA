"""
scoring.py
Multi-mechanism validation scorer, evidence grouper, and contradiction tracker.
Computes transparent rule-based overall validation scores, assigns ValidationStatus,
and formats the fixed Stage 11 feature dictionary.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .models import (
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
    ValidationResult,
)

VALIDATION_FEATURE_SCHEMA_VERSION = "validation_features_v1"


def evaluate_evidence_groups(
    crc_results: List[CRCResult],
    parity_results: List[ParityResult],
    syndrome_result: SyndromeResult,
    reencoding_result: ReencodingResult,
    repetition_result: FrameRepetitionResult,
    sync_results: List[SyncWordResult],
    header_results: List[HeaderConsistencyResult],
    length_result: LengthConsistencyResult,
    structural_result: StructuralResult,
) -> Tuple[int, int, int, List[str], Dict[str, float]]:
    """
    Evaluate independent evidence groups and identify contradictions.
    Returns:
        (raw_evidence_count, independent_group_count, strong_group_count, contradictions, group_scores)
    """
    contradictions = []
    group_scores = {
        "FEC_INTERNAL": 0.0,
        "CRC": 0.0,
        "FRAME_PERIODICITY": 0.0,
        "SYNC": 0.0,
        "HEADER_STRUCTURE": 0.0,
        "LENGTH_CONSISTENCY": 0.0,
        "REENCODING": 0.0,
        "GENERIC_STRUCTURE": 0.0,
    }

    raw_evidence_count = 0
    strong_group_count = 0
    independent_group_count = 0

    # 1. CRC Group
    if crc_results:
        best_crc = max(crc_results, key=lambda r: (1 if r.check_state == EvidenceCheckState.PASS else 0, r.pass_rate, r.width))
        if best_crc.checked_frames > 0:
            raw_evidence_count += 1
            if best_crc.check_state == EvidenceCheckState.PASS and best_crc.pass_rate >= 0.80:
                score = min(1.0, 0.7 + (best_crc.width / 32.0) * 0.3)
                group_scores["CRC"] = score
                if score >= 0.75:
                    strong_group_count += 1
                independent_group_count += 1
            elif best_crc.check_state == EvidenceCheckState.FAIL and best_crc.checked_frames >= 3:
                # Contradiction only if all checked CRC profiles failed
                all_failed = all(c.check_state == EvidenceCheckState.FAIL for c in crc_results if c.checked_frames >= 3)
                if all_failed:
                    contradictions.append(f"All tested CRC profiles failed across {best_crc.checked_frames} frames")

    # 2. FEC Internal Group
    if syndrome_result.syndrome_available:
        raw_evidence_count += 1
        if syndrome_result.check_state == EvidenceCheckState.PASS:
            group_scores["FEC_INTERNAL"] = syndrome_result.normalized_syndrome_score
            if syndrome_result.normalized_syndrome_score >= 0.80:
                strong_group_count += 1
            independent_group_count += 1
        elif syndrome_result.check_state == EvidenceCheckState.FAIL:
            contradictions.append("FEC syndrome / parity check failed")

    # 3. Frame Periodicity Group
    if repetition_result.repetition_count >= 2:
        raw_evidence_count += 1
        if repetition_result.check_state == EvidenceCheckState.PASS:
            score = repetition_result.periodicity_score
            group_scores["FRAME_PERIODICITY"] = min(1.0, score)
            if score >= 0.50 and repetition_result.repetition_count >= 3:
                strong_group_count += 1
            independent_group_count += 1

    # 4. Sync Word Group
    if sync_results:
        best_sync = max(sync_results, key=lambda s: s.periodicity_score if s.match_count > 0 else -1.0)
        if best_sync.match_count > 0:
            raw_evidence_count += 1
            if best_sync.check_state == EvidenceCheckState.PASS:
                score = (best_sync.periodicity_score * 0.7 + (1.0 - best_sync.mean_hamming_distance / 16.0) * 0.3)
                group_scores["SYNC"] = min(1.0, score)
                if best_sync.match_count >= 3 and best_sync.periodicity_score >= 0.75:
                    strong_group_count += 1
                independent_group_count += 1

    # 5. Header Structure Group
    if header_results:
        raw_evidence_count += len(header_results)
        passed_headers = [h for h in header_results if h.check_state == EvidenceCheckState.PASS]
        failed_headers = [h for h in header_results if h.check_state == EvidenceCheckState.FAIL]
        if passed_headers:
            avg_score = float(np.mean([h.consistency_score for h in passed_headers]))
            group_scores["HEADER_STRUCTURE"] = avg_score
            if avg_score >= 0.85 and len(passed_headers) >= 2:
                strong_group_count += 1
            independent_group_count += 1
        if failed_headers and len(failed_headers) > len(passed_headers):
            contradictions.append(f"Header consistency failed on {len(failed_headers)} fields")

    # 6. Length Consistency Group
    if length_result.check_state != EvidenceCheckState.NOT_TESTED:
        raw_evidence_count += 1
        if length_result.check_state == EvidenceCheckState.PASS:
            group_scores["LENGTH_CONSISTENCY"] = 1.0
            independent_group_count += 1
        else:
            contradictions.append(f"Declared length ({length_result.declared_length}) disagrees with observed ({length_result.observed_length})")

    # 7. Re-encoding Group
    if reencoding_result.check_state != EvidenceCheckState.NOT_TESTED:
        raw_evidence_count += 1
        if reencoding_result.check_state == EvidenceCheckState.PASS:
            group_scores["REENCODING"] = reencoding_result.bit_match_fraction
            if reencoding_result.bit_match_fraction >= 0.90:
                strong_group_count += 1
            independent_group_count += 1
        elif reencoding_result.check_state == EvidenceCheckState.FAIL:
            contradictions.append(f"Re-encoded bit mismatch high ({1.0 - reencoding_result.bit_match_fraction:.1%})")

    # 8. Generic Structure
    struct_score = max(0.0, 1.0 - abs(structural_result.binary_entropy - 0.95)) * 0.5 + structural_result.byte_alignment_score * 0.5
    group_scores["GENERIC_STRUCTURE"] = float(struct_score)
    if struct_score > 0.4:
        independent_group_count += 1

    return (
        raw_evidence_count,
        independent_group_count,
        strong_group_count,
        contradictions,
        group_scores,
    )


def compute_overall_validation_score(
    group_scores: Dict[str, float],
    contradiction_count: int,
    config_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute rule-based validation score normalized to [0, 1].
    """
    if config_weights is None:
        config_weights = {
            "crc_weight": 2.0,
            "syndrome_weight": 1.8,
            "reencoding_weight": 1.5,
            "sync_weight": 1.5,
            "repetition_weight": 1.2,
            "header_weight": 1.0,
            "length_weight": 1.0,
            "generic_structure_weight": 0.3,
            "contradiction_penalty": 1.5,
        }

    weighted_sum = (
        group_scores.get("CRC", 0.0) * config_weights.get("crc_weight", 2.0)
        + group_scores.get("FEC_INTERNAL", 0.0) * config_weights.get("syndrome_weight", 1.8)
        + group_scores.get("REENCODING", 0.0) * config_weights.get("reencoding_weight", 1.5)
        + group_scores.get("SYNC", 0.0) * config_weights.get("sync_weight", 1.5)
        + group_scores.get("FRAME_PERIODICITY", 0.0) * config_weights.get("repetition_weight", 1.2)
        + group_scores.get("HEADER_STRUCTURE", 0.0) * config_weights.get("header_weight", 1.0)
        + group_scores.get("LENGTH_CONSISTENCY", 0.0) * config_weights.get("length_weight", 1.0)
        + group_scores.get("GENERIC_STRUCTURE", 0.0) * config_weights.get("generic_structure_weight", 0.3)
    )

    penalty = contradiction_count * config_weights.get("contradiction_penalty", 1.5)
    raw_score = max(0.0, weighted_sum - penalty)

    # Definitive proofs (CRC, FEC syndrome, re-encoding) provide strong confidence anchors
    anchor_score = max(
        group_scores.get("CRC", 0.0),
        group_scores.get("FEC_INTERNAL", 0.0),
        group_scores.get("REENCODING", 0.0)
    )

    # Normalize by active tested evidence while respecting anchor proof
    total_possible = sum(config_weights.values()) - config_weights.get("contradiction_penalty", 1.5)
    normalized_score = min(1.0, max(anchor_score * 0.85, raw_score / max(1.0, total_possible * 0.35)))
    if contradiction_count > 0:
        normalized_score = max(0.0, normalized_score - 0.25 * contradiction_count)

    return float(round(normalized_score, 4))


def determine_validation_status(
    overall_score: float,
    strong_evidence_groups: int,
    independent_evidence_groups: int,
    contradiction_count: int,
    mode: str = "blind",
) -> ValidationStatus:
    """
    Decide validation status according to multi-mechanism evidence criteria.
    """
    if contradiction_count >= 2 and strong_evidence_groups == 0:
        return ValidationStatus.VALIDATION_FAILED

    if strong_evidence_groups >= 2 and contradiction_count == 0 and overall_score >= 0.70:
        return ValidationStatus.VALIDATION_STRONG

    if strong_evidence_groups >= 1 and contradiction_count <= 1 and overall_score >= 0.45:
        return ValidationStatus.VALIDATION_MODERATE

    if independent_evidence_groups >= 2 and overall_score >= 0.25:
        return ValidationStatus.VALIDATION_WEAK

    if contradiction_count > 0 and overall_score < 0.20:
        return ValidationStatus.VALIDATION_FAILED

    # Default fallback for blind signals without known protocol schemas
    return ValidationStatus.VALIDATION_INCONCLUSIVE


def extract_stage11_features(
    res: ValidationResult,
) -> Dict[str, float]:
    """
    Export fixed-order feature schema dictionary for downstream Stage 11 XGBoost Pipeline Scorer.
    """
    best_crc = None
    if res.crc_results:
        # Pick best passing CRC or highest pass rate
        best_crc = max(res.crc_results, key=lambda r: (1 if r.check_state == EvidenceCheckState.PASS else 0, r.pass_rate, r.width))

    best_sync = None
    if res.sync_word_results:
        best_sync = max(res.sync_word_results, key=lambda s: (1 if s.check_state == EvidenceCheckState.PASS else 0, s.periodicity_score, s.match_count))

    features = {
        "crc_available": 1.0 if best_crc and best_crc.checked_frames > 0 else 0.0,
        "crc_pass": 1.0 if best_crc and best_crc.check_state == EvidenceCheckState.PASS else 0.0,
        "crc_width": float(best_crc.width) if best_crc else 0.0,
        "crc_pass_rate": float(best_crc.pass_rate) if best_crc else 0.0,
        "parity_pass_rate": float(res.parity_results[0].pass_rate) if res.parity_results else 0.0,
        "syndrome_valid": 1.0 if res.syndrome_result.syndrome_valid else 0.0,
        "normalized_syndrome_score": float(res.syndrome_result.normalized_syndrome_score),
        "reencoding_match_fraction": float(res.reencoding_result.bit_match_fraction),
        "frame_periodicity_score": float(res.frame_repetition_result.periodicity_score),
        "frame_repetition_count": float(res.frame_repetition_result.repetition_count),
        "sync_match_count": float(best_sync.match_count) if best_sync else 0.0,
        "sync_periodicity_score": float(best_sync.periodicity_score) if best_sync else 0.0,
        "header_consistency_score": float(np.mean([h.consistency_score for h in res.header_results])) if res.header_results else 0.0,
        "length_consistency_score": 1.0 if res.length_result.is_valid and res.length_result.check_state != EvidenceCheckState.NOT_TESTED else 0.0,
        "binary_entropy": float(res.structural_result.binary_entropy),
        "bit_balance_zero": float(res.structural_result.bit_balance_zero),
        "byte_alignment_score": float(res.structural_result.byte_alignment_score),
        "independent_evidence_groups": float(res.evidence_count),
        "strong_evidence_groups": float(res.strong_evidence_count),
        "contradiction_count": float(res.contradiction_count),
        "overall_validation_score": float(res.overall_validation_score),
    }

    return features
