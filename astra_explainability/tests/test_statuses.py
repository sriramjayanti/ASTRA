"""
Unit tests for status assignments and field-specific rules.
Tests:
- TEST 8: CONFIRMED threshold requirement
- TEST 9: ESTIMATED threshold requirement
- TEST 10: POSSIBLE threshold requirement
- TEST 11: UNKNOWN threshold requirement
- TEST 12: Strong independent group requirement for CONFIRMED
- TEST 16: Field-specific statuses
"""

import pytest
from astra_explainability.src.models import AstraStatus
from astra_explainability.src.statuses import StatusEngine
from astra_explainability.src.inference import ExplainabilityEngine
from astra_explainability.src.utils import create_synthetic_perfect_record


def test_status_thresholds():
    engine = StatusEngine({
        "confirmed_threshold": 0.85,
        "estimated_threshold": 0.60,
        "possible_threshold": 0.35,
        "confirmed_min_strong_groups": 2
    })

    # TEST 8: Score >= 0.85 with >= 2 strong groups -> CONFIRMED
    status, reason = engine.determine_status(0.90, strong_groups=2, contradictions_count=0)
    assert status == AstraStatus.CONFIRMED

    # TEST 12: Score >= 0.85 with only 1 strong group -> Falls back to ESTIMATED!
    status_1grp, reason_1grp = engine.determine_status(0.92, strong_groups=1, contradictions_count=0)
    assert status_1grp == AstraStatus.ESTIMATED
    assert "insufficient independent strong evidence groups" in reason_1grp

    # TEST 9: Score between 0.60 and 0.85 -> ESTIMATED
    status_est, _ = engine.determine_status(0.72, strong_groups=1, contradictions_count=0)
    assert status_est == AstraStatus.ESTIMATED

    # TEST 10: Score between 0.35 and 0.60 -> POSSIBLE
    status_pos, _ = engine.determine_status(0.48, strong_groups=0, contradictions_count=0)
    assert status_pos == AstraStatus.POSSIBLE

    # TEST 11: Score < 0.35 -> UNKNOWN
    status_unk, _ = engine.determine_status(0.20, strong_groups=0, contradictions_count=0)
    assert status_unk == AstraStatus.UNKNOWN


def test_16_field_specific_statuses():
    """Ensure different fields hold different honest statuses instead of one flat global status."""
    engine = ExplainabilityEngine()
    record = create_synthetic_perfect_record()
    res = engine.explain(record)

    # Modulation and FEC are CONFIRMED
    assert res.field_explanations["modulation"].status == AstraStatus.CONFIRMED
    assert res.field_explanations["fec"].status == AstraStatus.CONFIRMED
    assert res.field_explanations["payload"].status == AstraStatus.CONFIRMED

    # Protocol profile is None in synthetic record -> UNKNOWN
    assert res.field_explanations["protocol"].status == AstraStatus.UNKNOWN

    # Text interpretation is ESTIMATED
    assert res.field_explanations["text_representation"].status == AstraStatus.ESTIMATED
