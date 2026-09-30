"""
test_entropy.py
Unit tests for binary entropy, block entropy, sliding-window entropy, and change-point detection.
"""

import pytest
import numpy as np
from astra_bitstream_intelligence.src.entropy import (
    binary_entropy,
    block_entropy,
    compute_all_ngram_entropies,
    sliding_entropy,
    detect_entropy_change_points
)


def test_1_binary_entropy_all_zeros():
    """TEST 1: binary entropy of all zeros = 0."""
    zeros = np.zeros(100, dtype=np.uint8)
    h = binary_entropy(zeros)
    assert h == 0.0

    ones = np.ones(100, dtype=np.uint8)
    h_ones = binary_entropy(ones)
    assert h_ones == 0.0


def test_2_balanced_sequence_entropy():
    """TEST 2: balanced random-ish sequence entropy near 1."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, size=1000, dtype=np.uint8)
    h = binary_entropy(bits)
    assert 0.95 <= h <= 1.0


def test_3_block_entropy_calculation():
    """TEST 3: block entropy calculation for n=1, 2, 4, 8."""
    # Repeating alternating sequence [1, 0, 1, 0] has 1-bit entropy 1.0, but 2-bit entropy 0.0
    bits = np.tile(np.array([1, 0], dtype=np.uint8), 50)
    h1 = block_entropy(bits, ngram_size=1)
    h2 = block_entropy(bits, ngram_size=2)
    assert abs(h1 - 1.0) < 1e-4
    assert h2 == 0.0 # Only symbol "10" appears, 0 entropy

    # High entropy block test
    rng = np.random.RandomState(42)
    rand_bits = rng.randint(0, 2, size=2048, dtype=np.uint8)
    ngram_dict = compute_all_ngram_entropies(rand_bits, ngram_sizes=[1, 2, 4, 8])
    for k, v in ngram_dict.items():
        assert 0.85 <= v <= 1.0


def test_4_sliding_entropy_and_change_points():
    """TEST 4: sliding entropy output length and change-point detection."""
    # Construct sequence with low-entropy prefix, high-entropy payload, low-entropy suffix
    prefix = np.zeros(256, dtype=np.uint8)
    rng = np.random.RandomState(42)
    payload = rng.randint(0, 2, size=512, dtype=np.uint8)
    suffix = np.ones(256, dtype=np.uint8)
    combined = np.concatenate((prefix, payload, suffix))

    win_size = 128
    step = 16
    windows = sliding_entropy(combined, window_size=win_size, step=step)

    expected_len = (len(combined) - win_size) // step + 1
    assert len(windows) == expected_len

    change_points = detect_entropy_change_points(windows, threshold=0.15, step=step)
    assert len(change_points) > 0
