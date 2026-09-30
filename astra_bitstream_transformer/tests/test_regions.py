"""
test_regions.py
Unit tests for contiguous region extraction, small-region filtering, and UNKNOWN thresholding.
"""

import pytest
import numpy as np
from astra_bitstream_transformer.src.models import StructureLabel
from astra_bitstream_transformer.src.postprocess import (
    extract_contiguous_regions,
    filter_small_regions,
    apply_unknown_confidence_threshold
)


def test_15_region_extraction():
    """TEST 15: contiguous region extraction from per-bit labels."""
    # 0..31: SYNC (1), 32..95: HEADER (2), 96..200: PAYLOAD (3)
    labels = np.array([1] * 32 + [2] * 64 + [3] * 105, dtype=np.int64)
    probs = np.zeros((len(labels), 6), dtype=np.float32)
    for i, l in enumerate(labels):
        probs[i, l] = 0.95

    regions = extract_contiguous_regions(labels, probs)
    assert len(regions) == 3

    assert regions[0].label == "SYNC"
    assert regions[0].start_bit == 0
    assert regions[0].end_bit == 32
    assert regions[0].bit_length == 32

    assert regions[1].label == "HEADER"
    assert regions[1].start_bit == 32
    assert regions[1].end_bit == 96

    assert regions[2].label == "PAYLOAD"
    assert regions[2].start_bit == 96
    assert regions[2].end_bit == 201


def test_16_small_region_filtering():
    """TEST 16: small 1-bit or 2-bit spike regions are filtered and merged."""
    # 0..31: SYNC, 32: single-bit PAYLOAD spike, 33..95: HEADER
    labels = np.array([1] * 32 + [3] * 1 + [2] * 63, dtype=np.int64)
    probs = np.zeros((len(labels), 6), dtype=np.float32)
    for i, l in enumerate(labels):
        probs[i, l] = 0.90

    raw_regions = extract_contiguous_regions(labels, probs)
    assert len(raw_regions) == 3

    cleaned = filter_small_regions(raw_regions, min_region_sizes={"PAYLOAD": 16})
    assert len(cleaned) == 2 # Spike merged


def test_17_unknown_threshold():
    """TEST 17: low confidence positions are mapped to UNKNOWN (0)."""
    labels = np.array([1, 1, 2, 2], dtype=np.int64)
    # Third bit has low max probability (0.30 < 0.40)
    probs = np.array([
        [0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.8, 0.0, 0.1, 0.0, 0.0],
        [0.2, 0.2, 0.3, 0.1, 0.1, 0.1], # Low confidence
        [0.0, 0.0, 0.9, 0.0, 0.0, 0.1]
    ], dtype=np.float32)

    refined = apply_unknown_confidence_threshold(labels, probs, unknown_threshold=0.40)
    assert refined[2] == StructureLabel.UNKNOWN.value
    assert refined[0] == StructureLabel.SYNC.value
    assert refined[3] == StructureLabel.HEADER.value
