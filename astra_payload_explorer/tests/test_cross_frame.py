import pytest
import numpy as np
import json
from astra_payload_explorer.src.models import (
    FrameRecord,
    RegionInfo,
    PayloadViews,
    PayloadExplorerResult,
    ExplorerStatus,
    EvidenceLevel
)
from astra_payload_explorer.src.cross_frame import (
    detect_duplicate_frames,
    analyze_payload_variations,
    check_sequence_continuity
)
from astra_payload_explorer.src.exporters import export_to_json, export_to_csv, generate_hex_dump

def test_duplicate_detection():
    # 3 frames: frame 0 and frame 2 are identical, frame 1 differs
    p_bits_a = np.array([1, 0, 1, 0] * 8, dtype=np.uint8)
    p_bits_b = np.array([0, 1, 0, 1] * 8, dtype=np.uint8)

    f0 = FrameRecord(
        frame_index=0,
        start_bit=0,
        end_bit=32,
        frame_length_bits=32,
        frame_hash="hash_a",
        payload_hash="phash_a",
        payload=PayloadViews(raw_bits="1010"*8, hex_str="aaaa", length_bits=32)
    )
    f1 = FrameRecord(
        frame_index=1,
        start_bit=32,
        end_bit=64,
        frame_length_bits=32,
        frame_hash="hash_b",
        payload_hash="phash_b",
        payload=PayloadViews(raw_bits="0101"*8, hex_str="5555", length_bits=32)
    )
    f2 = FrameRecord(
        frame_index=2,
        start_bit=64,
        end_bit=96,
        frame_length_bits=32,
        frame_hash="hash_a",
        payload_hash="phash_a",
        payload=PayloadViews(raw_bits="1010"*8, hex_str="aaaa", length_bits=32)
    )

    dupes = detect_duplicate_frames([f0, f1, f2])
    assert "hash_a" in dupes
    assert dupes["hash_a"] == [0, 2]
    assert "hash_b" not in dupes

def test_json_and_csv_serialization():
    f0 = FrameRecord(
        frame_index=0,
        start_bit=0,
        end_bit=64,
        frame_length_bits=64,
        frame_hash="abc123hash",
        frame_quality_score=0.98,
        payload=PayloadViews(
            length_bits=32,
            length_bytes=4,
            hex_str="68692020",
            raw_bits="01101000011010010010000000100000",
            byte_array=[0x68, 0x69, 0x20, 0x20],
            representations={"utf8": {"valid": True, "value": "hi  "}}
        )
    )

    res = PayloadExplorerResult(
        pipeline_path_id="pipe_test_01",
        frame_count=1,
        frames=[f0],
        selected_frame_length=64,
        explorer_status=ExplorerStatus.STRUCTURE_PARSED,
        candidate_interpretations=[]
    )

    json_str = export_to_json(res)
    loaded = json.loads(json_str)
    assert loaded["pipeline_path_id"] == "pipe_test_01"
    assert loaded["frame_count"] == 1
    assert loaded["frames"][0]["frame_hash"] == "abc123hash"
    assert loaded["frames"][0]["payload"]["hex_str"] == "68692020"

    # Hex dump test
    dump = generate_hex_dump(bytes([0x68, 0x69, 0x20, 0x20]))
    assert "68 69 20 20" in dump
    assert "hi  " in dump
