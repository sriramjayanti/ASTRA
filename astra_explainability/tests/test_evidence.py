"""
Unit tests for EvidenceItem creation, extraction, and deduplication.
Tests:
- TEST 1: EvidenceItem creation & normalization
- TEST 2: Evidence groups assignment
- TEST 3: Independence deduplication & group counts
- TEST 4: Supporting evidence filtering
- TEST 5: Contradictory evidence filtering
- TEST 22: Missing CRC != CRC fail
- TEST 23: User override classification
"""

import pytest
from astra_explainability.src.models import (
    EvidenceItem, EvidenceCategory, EvidenceStrength,
    EvidenceDirection, IndependenceGroup
)
from astra_explainability.src.evidence import EvidenceExtractor
from astra_explainability.src.evidence_groups import EvidenceGrouper
from astra_explainability.src.utils import create_synthetic_perfect_record, create_user_override_record, create_missing_crc_record


def test_1_evidence_item_creation():
    item = EvidenceItem(
        evidence_id="ev_001",
        stage=3,
        category=EvidenceCategory.MODEL_PREDICTION,
        description="1D ResNet QPSK probability 0.84",
        value=0.84,
        normalized_score=0.84,
        strength=EvidenceStrength.STRONG,
        direction=EvidenceDirection.SUPPORT,
        independence_group=IndependenceGroup.MODULATION_MODELS,
        source_reference="stage_3_fusion.1d_resnet"
    )
    assert item.evidence_id == "ev_001"
    assert item.stage == 3
    assert item.normalized_score == 0.84
    assert item.strength == EvidenceStrength.STRONG
    assert item.direction == EvidenceDirection.SUPPORT
    assert item.independence_group == IndependenceGroup.MODULATION_MODELS


def test_2_evidence_groups_assignment():
    record = create_synthetic_perfect_record()
    extractor = EvidenceExtractor()
    evidence_list = extractor.extract_all(record)

    groups = {ev.independence_group for ev in evidence_list}
    assert IndependenceGroup.MODULATION_MODELS in groups
    assert IndependenceGroup.CONSTELLATION in groups
    assert IndependenceGroup.SYNC in groups
    assert IndependenceGroup.CRC in groups
    assert IndependenceGroup.FEC_INTERNAL in groups
    assert IndependenceGroup.PAYLOAD_STRUCTURE in groups


def test_3_independence_deduplication():
    grouper = EvidenceGrouper()

    # Two items in the same independence group (CRC)
    crc_pass = EvidenceItem(
        evidence_id="crc_1", stage=10, category=EvidenceCategory.CRC_VALIDATION,
        description="8/8 CRC checks passed", value=8, normalized_score=1.0,
        strength=EvidenceStrength.STRONG, direction=EvidenceDirection.SUPPORT,
        independence_group=IndependenceGroup.CRC
    )
    val_score = EvidenceItem(
        evidence_id="val_1", stage=10, category=EvidenceCategory.CRC_VALIDATION,
        description="Downstream validation score 0.98", value=0.98, normalized_score=0.98,
        strength=EvidenceStrength.STRONG, direction=EvidenceDirection.SUPPORT,
        independence_group=IndependenceGroup.CRC
    )

    items = [crc_pass, val_score]
    # Even though there are 2 strong items, they belong to the SAME group: CRC
    strong_group_count = grouper.count_strong_independent_groups(items)
    assert strong_group_count == 1, "Correlated CRC items must not be counted as multiple independent groups"

    grouped = grouper.group_evidence(items)
    assert IndependenceGroup.CRC in grouped
    assert len(grouped[IndependenceGroup.CRC]) == 2


def test_4_support_evidence_filtering():
    record = create_synthetic_perfect_record()
    extractor = EvidenceExtractor()
    items = extractor.extract_all(record)

    support_items = [e for e in items if e.direction == EvidenceDirection.SUPPORT]
    assert len(support_items) > 5
    assert any("QPSK" in e.description for e in support_items)
    assert any("CRC" in e.description for e in support_items)


def test_5_contradictory_evidence():
    record = create_synthetic_perfect_record()
    extractor = EvidenceExtractor()
    items = extractor.extract_all(record)

    contradict_items = [e for e in items if e.direction == EvidenceDirection.CONTRADICT]
    # In perfect record, secondary candidates like 8PSK are contradictory/competing
    assert len(contradict_items) >= 1
    assert any("8PSK" in e.description for e in contradict_items)


def test_22_missing_crc_not_crc_fail():
    record = create_missing_crc_record()
    extractor = EvidenceExtractor()
    items = extractor.extract_all(record)

    # When total_checked == 0, it is not a CRC failure, it's missing/untested
    crc_failures = [e for e in items if e.category == EvidenceCategory.CRC_VALIDATION and e.direction == EvidenceDirection.CONTRADICT]
    assert len(crc_failures) == 0, "Missing CRC checks must not be labeled as failed CRC checks"


def test_23_user_override_classification():
    record = create_user_override_record()
    extractor = EvidenceExtractor()
    items = extractor.extract_all(record)

    override_items = [e for e in items if e.category == EvidenceCategory.USER_OVERRIDE]
    assert len(override_items) == 1
    ov = override_items[0]
    assert ov.independence_group == IndependenceGroup.USER_OVERRIDE
    assert ov.value == "16QAM"
    assert "User override" in ov.description
