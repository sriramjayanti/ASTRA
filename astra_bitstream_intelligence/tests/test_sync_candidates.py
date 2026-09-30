"""
test_sync_candidates.py
Unit tests for known sync detection, blind sync discovery, and Hamming-tolerant matching.
"""

import pytest
import numpy as np
from astra_bitstream_intelligence.src.utils import generate_synthetic_framed_bitstream
from astra_bitstream_intelligence.src.cross_correlation import search_known_sync
from astra_bitstream_intelligence.src.sync_candidates import discover_candidate_sync
from astra_bitstream_intelligence.src.repeated_patterns import analyze_repeated_prefixes
from astra_bitstream_intelligence.src.models import PeriodicityCandidate


def test_11_repeated_prefix_detected():
    """TEST 11: repeated prefix detected across candidate frames."""
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=256,
        num_frames=16,
        sync_word_hex="EB90",
        seed=42
    )

    prefixes = analyze_repeated_prefixes(bits, frame_length=256, prefix_lengths=[16])
    assert len(prefixes) > 0
    # EB90 in binary is: 1110101110010000
    assert prefixes[0].pattern_bits == "1110101110010000"
    assert prefixes[0].occurrence_count == 16


def test_13_known_sync_detected():
    """TEST 13: known sync detected at exact positions."""
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=10,
        sync_word_hex="EB90",
        seed=123
    )

    sr = search_known_sync(bits, pattern="EB90")
    assert sr.match_count == 10
    assert sr.positions == [i * 512 for i in range(10)]
    assert sr.confidence > 0.85


def test_14_hamming_tolerant_sync():
    """TEST 14: Hamming-tolerant sync matching with bit errors."""
    # 2% BER injected
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=10,
        sync_word_hex="1ACFFC1D",
        ber=0.03,
        seed=456
    )

    sr = search_known_sync(bits, pattern="1ACFFC1D", max_hamming_fraction=0.15)
    assert sr.match_count >= 8 # High recall despite channel errors


def test_15_sync_spacing_measured():
    """TEST 15: sync spacing measured with low variance."""
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=8,
        sync_word_hex="EB90",
        seed=789
    )

    sr = search_known_sync(bits, pattern="EB90")
    assert sr.mean_spacing == 512.0
    assert sr.spacing_variance == 0.0
    assert sr.periodicity_score == 1.0


def test_90_soft_correlation_sync():
    """TEST 90: soft correlation sync matching."""
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=256,
        num_frames=6,
        sync_word_hex="EB90",
        seed=321
    )
    # Generate soft LLRs with high confidence
    soft = (2.0 * bits.astype(np.float32) - 1.0) * 3.5

    sr = search_known_sync(bits, pattern="EB90", soft_info=soft)
    assert sr.match_count == 6
    assert all(score >= 0.90 for score in sr.match_scores)
