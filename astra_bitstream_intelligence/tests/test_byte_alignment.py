"""
test_byte_alignment.py
Unit tests for 8-bit offset byte alignment exploration.
"""

import pytest
import numpy as np
from astra_bitstream_intelligence.src.byte_alignment import (
    bits_to_bytes,
    compute_byte_entropy,
    analyze_byte_offsets
)


def test_19_byte_offset_analysis():
    """TEST 19: byte-offset analysis tests all 8 offsets."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, size=512, dtype=np.uint8)

    res = analyze_byte_offsets(bits)
    assert 0 <= res.best_offset <= 7
    assert len(res.offset_entropies) == 8
    assert len(res.zero_byte_fractions) == 8
    assert len(res.structural_scores) == 8


def test_92_byte_alignment_offset_detection():
    """TEST 92: embedded byte structure at known bit offset is recovered."""
    # Create byte structure with many 0x00 bytes and low byte entropy at offset 5
    offset = 5
    prefix = np.zeros(offset, dtype=np.uint8)

    # 100 bytes where every 4th byte is 0x00 and others are from small alphabet
    structured_bytes = []
    for i in range(120):
        if i % 4 == 0:
            structured_bytes.append(0x00)
        else:
            structured_bytes.append(0x55)

    # Convert bytes to bits
    struct_bits = []
    for b in structured_bytes:
        for s in (7, 6, 5, 4, 3, 2, 1, 0):
            struct_bits.append((b >> s) & 1)

    combined_bits = np.concatenate((prefix, np.array(struct_bits, dtype=np.uint8)))

    res = analyze_byte_offsets(combined_bits)
    assert res.best_offset == offset
    assert res.zero_byte_fractions[offset] > 0.20
