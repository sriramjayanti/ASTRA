"""
test_header_consistency.py
Unit tests for header consistency rules (constant, monotonic sequence counter).
"""

import pytest
import numpy as np
from astra_validation.src.headers import (
    validate_constant_field,
    validate_monotonic_counter,
    validate_header_schema,
)
from astra_validation.src.models import EvidenceCheckState


def test_constant_header_field():
    frames = []
    # Version = 2 (0010) in bits 0..3
    for _ in range(5):
        f = np.zeros(64, dtype=np.uint8)
        f[2] = 1 # 0010 = 2
        frames.append(f)

    res = validate_constant_field(frames, "version", offset_bits=0, length_bits=4, expected_value=2)
    assert res.frames_tested == 5
    assert res.frames_valid == 5
    assert res.consistency_score == 1.0
    assert res.check_state == EvidenceCheckState.PASS


def test_monotonic_sequence_counter():
    frames = []
    # Counter 10, 11, 12, 13 at offset 4, length 8
    for i in range(4):
        f = np.zeros(64, dtype=np.uint8)
        val = 10 + i
        for b in range(8):
            f[4 + b] = (val >> (7 - b)) & 1
        frames.append(f)

    res = validate_monotonic_counter(frames, "sequence_counter", offset_bits=4, length_bits=8)
    assert res.frames_tested == 4
    assert res.consistency_score == 1.0
    assert res.check_state == EvidenceCheckState.PASS
