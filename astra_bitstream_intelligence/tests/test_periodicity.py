"""
test_periodicity.py
Unit tests for harmonic merging and multi-period analysis.
"""

import pytest
from astra_bitstream_intelligence.src.models import AutocorrPeak
from astra_bitstream_intelligence.src.periodicity import (
    is_harmonic,
    merge_harmonics,
    rank_period_candidates
)


def test_8_harmonic_merging():
    """TEST 8: harmonic merging clusters harmonics under fundamental."""
    peaks = [
        AutocorrPeak(lag=512, correlation=0.82),
        AutocorrPeak(lag=1024, correlation=0.74),
        AutocorrPeak(lag=1536, correlation=0.65),
        AutocorrPeak(lag=200, correlation=0.30)
    ]

    candidates = merge_harmonics(peaks, tolerance=0.05)
    assert len(candidates) >= 1

    # Fundamental 512 should be top candidate with harmonic support
    top_cand = candidates[0]
    assert top_cand.period_bits == 512
    assert "harmonic" in top_cand.support_sources[0]


def test_87_multi_period_ranking():
    """TEST 87: multi-period ranking handles subharmonics with sync guidance."""
    peaks = [
        AutocorrPeak(lag=64, correlation=0.75),
        AutocorrPeak(lag=512, correlation=0.80),
        AutocorrPeak(lag=1024, correlation=0.70)
    ]

    candidates = merge_harmonics(peaks)
    # Re-rank with sync spacing guidance of 512 bits
    ranked = rank_period_candidates(candidates, sync_spacings=[512.0])

    top_periods = [r.period_bits for r in ranked]
    assert 512 in top_periods
    # 512 candidate should have received sync_spacing boost
    cand_512 = next(c for c in ranked if c.period_bits == 512)
    assert "sync_spacing" in cand_512.support_sources
