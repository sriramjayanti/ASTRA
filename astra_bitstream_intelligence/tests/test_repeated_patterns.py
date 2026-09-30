"""
test_repeated_patterns.py
Unit tests for recurring bit-pattern discovery.
"""

import pytest
import numpy as np
from astra_bitstream_intelligence.src.repeated_patterns import find_repeated_patterns


def test_16_repeated_patterns_detected():
    """TEST 16: repeated bit sequences detected accurately."""
    # Pattern of length 16 repeated at periodic offsets
    pattern = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    rng = np.random.RandomState(42)

    all_bits = []
    for _ in range(8):
        all_bits.extend(pattern)
        all_bits.extend(rng.randint(0, 2, size=100, dtype=np.uint8))

    bits = np.array(all_bits, dtype=np.uint8)
    repeated = find_repeated_patterns(bits, pattern_lengths=[16], min_occurrences=4)

    assert len(repeated) > 0
    top_pat = repeated[0]
    assert top_pat.length == 16
    assert top_pat.occurrence_count >= 8
