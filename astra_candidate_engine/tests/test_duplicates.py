"""
test_duplicates.py
Tests for duplicate symbol-rate candidate merging within tolerance.
"""

from astra_candidate_engine.src.pruning import merge_duplicate_rates


def test_rate_deduplication():
    rates = [
        {"symbol_rate_hz": 9600.0, "score": 0.85},
        {"symbol_rate_hz": 9580.0, "score": 0.60},
        {"symbol_rate_hz": 4800.0, "score": 0.15},
    ]
    merged = merge_duplicate_rates(rates, tolerance_percent=2.0)
    assert len(merged) == 2
    assert merged[0]["symbol_rate_hz"] == 9600.0
