"""
test_sync_quality.py
Unit tests for constellation compactness and lock metrics.
"""

import numpy as np
from astra_synchronization.src.quality import calculate_constellation_compactness, calculate_sync_lock_metrics, evaluate_sync_status


def test_quality_compactness_and_metrics():
    # Compact clean QPSK
    clean_qpsk = (np.array([1+1j, 1-1j, -1+1j, -1-1j] * 50) / np.sqrt(2.0)).astype(np.complex64)
    q_clean = calculate_constellation_compactness(clean_qpsk)
    
    # Dispersed noisy points
    noisy = clean_qpsk + 0.8 * (np.random.randn(len(clean_qpsk)) + 1j * np.random.randn(len(clean_qpsk)))
    q_noisy = calculate_constellation_compactness(noisy)
    
    assert q_clean > q_noisy
    
    overall, metrics = calculate_sync_lock_metrics(
        timing_lock=0.9, carrier_lock=0.85, frequency_lock=0.95, const_quality_before=0.5, const_quality_after=0.9
    )
    assert overall > 0.70
    status, passed, reason = evaluate_sync_status(overall, timing_lock=0.9, carrier_lock=0.85)
    assert passed is True
    assert status == "SYNC_PASSED"
