"""
test_length_consistency.py
Unit tests for length-field validation vs observed boundaries.
"""

import pytest
import numpy as np
from astra_validation.src.length_checks import validate_length_field
from astra_validation.src.models import EvidenceCheckState


def test_length_field_match():
    # 512-bit frame (64 bytes total)
    # Header overhead = 6 bytes, CRC overhead = 2 bytes -> declared payload = 56 bytes
    frames = []
    for _ in range(4):
        f = np.zeros(512, dtype=np.uint8)
        # Declared length = 56 at offset 32, length 8
        val = 56
        for b in range(8):
            f[32 + b] = (val >> (7 - b)) & 1
        frames.append(f)

    res = validate_length_field(
        frames,
        offset_bits=32,
        length_bits=8,
        unit="bytes",
        observed_frame_length_bits=512,
        header_overhead_bytes=6,
        crc_overhead_bytes=2,
    )
    assert res.is_valid is True
    assert res.declared_length == 56
    assert res.observed_length == 64
    assert res.check_state == EvidenceCheckState.PASS


def test_length_field_mismatch():
    frames = []
    for _ in range(4):
        f = np.zeros(512, dtype=np.uint8)
        # Declared length = 200 (impossible for 64-byte frame)
        val = 200
        for b in range(8):
            f[32 + b] = (val >> (7 - b)) & 1
        frames.append(f)

    res = validate_length_field(
        frames,
        offset_bits=32,
        length_bits=8,
        unit="bytes",
        observed_frame_length_bits=512,
    )
    assert res.is_valid is False
    assert res.check_state == EvidenceCheckState.FAIL
