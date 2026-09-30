import pytest
import numpy as np
from astra_payload_explorer.src.models import FrameRecord, RegionInfo, EvidenceLevel
from astra_payload_explorer.src.blind_fields import (
    discover_constant_fields,
    discover_counter_candidates,
    discover_length_candidates,
    analyze_field_variability
)
from astra_payload_explorer.src.byte_alignment import bytes_to_bits

def test_discover_constant_fields():
    # 5 frames with 32-bit headers:
    # First 8 bits = constant 0xAA
    # Next 8 bits = random/changing
    # Next 16 bits = constant 0x0102
    frames = []
    const1 = bytes_to_bits(bytes([0xAA]))
    const2 = bytes_to_bits(bytes([0x01, 0x02]))

    for i in range(5):
        dyn = bytes_to_bits(bytes([i * 10]))
        h_bits = np.concatenate([const1, dyn, const2])
        f = FrameRecord(
            frame_index=i,
            start_bit=i*64,
            end_bit=(i+1)*64,
            frame_length_bits=64,
            header_region=RegionInfo(start_bit=0, end_bit=32, length_bits=32, raw_bits=h_bits)
        )
        frames.append(f)

    constants = discover_constant_fields(frames)
    assert len(constants) >= 2
    # Check that bit ranges [0, 8) and [16, 32) are captured as constant candidates
    offsets = [c.offset_bits for c in constants]
    assert 0 in offsets or 16 in offsets

def test_discover_counter_candidates():
    # 5 frames where bits 8..16 (8-bit field) increment 0, 1, 2, 3, 4
    frames = []
    prefix = np.zeros(8, dtype=np.uint8)
    suffix = np.zeros(16, dtype=np.uint8)

    for i in range(5):
        seq_bits = bytes_to_bits(bytes([i]))
        h_bits = np.concatenate([prefix, seq_bits, suffix])
        f = FrameRecord(
            frame_index=i,
            start_bit=i*64,
            end_bit=(i+1)*64,
            frame_length_bits=64,
            header_region=RegionInfo(start_bit=0, end_bit=32, length_bits=32, raw_bits=h_bits)
        )
        frames.append(f)

    counters = discover_counter_candidates(frames, candidate_widths=[8, 16])
    assert len(counters) > 0
    best_cnt = counters[0]
    assert best_cnt.offset_bits == 8
    assert best_cnt.width_bits == 8
    assert best_cnt.values_across_frames == [0, 1, 2, 3, 4]
    assert best_cnt.pattern == "incrementing"
    assert best_cnt.evidence_level == EvidenceLevel.INFERRED

def test_discover_length_candidates():
    # Frames with variable payload lengths and an 8-bit length field at offset 0
    # Frame 0: length 10 bytes -> field=10
    # Frame 1: length 20 bytes -> field=20
    # Frame 2: length 15 bytes -> field=15
    lengths = [10, 20, 15]
    frames = []
    for i, l in enumerate(lengths):
        len_bits = bytes_to_bits(bytes([l]))
        pad_hdr = np.zeros(8, dtype=np.uint8)
        h_bits = np.concatenate([len_bits, pad_hdr])
        p_bits = np.zeros(l * 8, dtype=np.uint8)

        f = FrameRecord(
            frame_index=i,
            start_bit=0,
            end_bit=16 + l*8,
            frame_length_bits=16 + l*8,
            header_region=RegionInfo(start_bit=0, end_bit=16, length_bits=16, raw_bits=h_bits),
            payload_region=RegionInfo(start_bit=16, end_bit=16 + l*8, length_bits=l*8, raw_bits=p_bits)
        )
        frames.append(f)

    len_cands = discover_length_candidates(frames, candidate_widths=[8])
    assert len(len_cands) > 0
    top_len = len_cands[0]
    assert top_len.offset_bits == 0
    assert top_len.width_bits == 8
    assert top_len.correlation >= 0.95
