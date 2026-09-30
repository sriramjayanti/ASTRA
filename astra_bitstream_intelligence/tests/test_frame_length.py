"""
test_frame_length.py
Unit tests for frame-length estimation, offset detection, and changing payload tolerance.
"""

import pytest
import numpy as np
from astra_bitstream_intelligence.src.utils import generate_synthetic_framed_bitstream
from astra_bitstream_intelligence.src.autocorrelation import bit_autocorrelation, find_autocorrelation_peaks
from astra_bitstream_intelligence.src.periodicity import merge_harmonics, rank_period_candidates
from astra_bitstream_intelligence.src.frame_length import estimate_frame_lengths
from astra_bitstream_intelligence.src.segmentation import find_optimal_frame_offset


def test_9_frame_length_512_detected():
    """TEST 9: frame-length 512 detected correctly."""
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=20,
        sync_word_hex="EB90",
        header_length=32,
        seed=42
    )

    corr = bit_autocorrelation(bits, max_lag_limit=2048)
    peaks = find_autocorrelation_peaks(corr, min_lag=32, peak_threshold=0.05)
    period_cands = merge_harmonics(peaks)

    frame_cands = estimate_frame_lengths(bits, period_cands, top_k=5)
    assert len(frame_cands) > 0
    top_cand = frame_cands[0]
    assert top_cand.period_bits == 512
    assert top_cand.score >= 0.50


def test_10_frame_offset_detected():
    """TEST 10: frame offset detected when stream is misaligned by noise prefix."""
    prefix_noise = 27
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=256,
        num_frames=16,
        sync_word_hex="EB90",
        prefix_noise_bits=prefix_noise,
        seed=101
    )

    best_offset, score = find_optimal_frame_offset(bits, frame_length=256, max_search_offset=64)
    assert best_offset == prefix_noise


def test_12_changing_payload_tolerated():
    """TEST 12: changing random payload each frame is tolerated and frame length is detected."""
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=30,
        sync_word_hex="1ACFFC1D",
        header_length=32,
        seed=999
    )

    corr = bit_autocorrelation(bits, max_lag_limit=2048)
    peaks = find_autocorrelation_peaks(corr, min_lag=32, peak_threshold=0.05)
    period_cands = merge_harmonics(peaks)

    frame_cands = estimate_frame_lengths(bits, period_cands, top_k=5)
    assert len(frame_cands) > 0
    assert frame_cands[0].period_bits == 512
