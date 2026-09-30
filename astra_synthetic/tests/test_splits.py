"""
Unit tests for train/validation/test splitting and leakage prevention (Engine 8).
"""

from pathlib import Path
import pytest
from astra_synthetic.payload.validators import ValidationError
from astra_synthetic.orchestration.models import DatasetRecord
from astra_synthetic.orchestration.splits import assign_split_by_chain_id, verify_no_split_leakage


def test_split_assignment_deterministic():
    chain_1 = "chain_00000001"
    sp1 = assign_split_by_chain_id(chain_1)
    sp2 = assign_split_by_chain_id(chain_1)
    assert sp1 == sp2
    assert sp1 in ("train", "validation", "test")


def test_split_leakage_detector():
    # Construct disjoint records
    r1 = DatasetRecord(
        dataset_record_id="ds_1",
        source_chain_id="chain_1",
        payload_id="p1",
        frame_id="f1",
        fec_record_id="fec1",
        interleaver_record_id="i1",
        modulation_record_id="m1",
        channel_record_id="c1",
        capture_record_id="cap1",
        capture_path="c1.iq",
        visible_metadata_path="c1.json",
        truth_metadata_path="t1.json",
        split="train",
    )
    r2 = DatasetRecord(
        dataset_record_id="ds_2",
        source_chain_id="chain_2",
        payload_id="p2",
        frame_id="f2",
        fec_record_id="fec2",
        interleaver_record_id="i2",
        modulation_record_id="m2",
        channel_record_id="c2",
        capture_record_id="cap2",
        capture_path="c2.iq",
        visible_metadata_path="c2.json",
        truth_metadata_path="t2.json",
        split="test",
    )

    is_valid, report = verify_no_split_leakage([r1, r2])
    assert is_valid is True
    assert report["leakage_detected"] is False

    # Simulate leakage: same source_chain_id 'chain_1' appears in test split
    r3 = DatasetRecord(
        dataset_record_id="ds_3",
        source_chain_id="chain_1",  # Same chain in another split!
        payload_id="p1",
        frame_id="f1",
        fec_record_id="fec1",
        interleaver_record_id="i1",
        modulation_record_id="m1",
        channel_record_id="c1",
        capture_record_id="cap3",
        capture_path="c3.iq",
        visible_metadata_path="c3.json",
        truth_metadata_path="t3.json",
        split="test",
    )

    with pytest.raises(ValidationError):
        verify_no_split_leakage([r1, r2, r3])
