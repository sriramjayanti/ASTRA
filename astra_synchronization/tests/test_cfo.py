"""
test_cfo.py
Unit tests for CFO estimation and correction.
"""

from astra_synchronization.src.cfo import estimate_cfo, correct_cfo, estimate_residual_cfo
from astra_synchronization.src.utils import generate_synthetic_test_signal


def test_cfo_recovery_accuracy():
    true_cfo = 1450.0
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", cfo_hz=true_cfo, sample_rate_hz=192000.0, snr_db=30.0
    )
    cfo_est, _ = estimate_cfo(iq, sample_rate_hz=192000.0, modulation="QPSK", modulation_family="PSK")
    assert abs(cfo_est - true_cfo) < 100.0
    
    corrected = correct_cfo(iq, cfo_hz=cfo_est, sample_rate_hz=192000.0)
    res_cfo = estimate_residual_cfo(corrected, sample_rate_hz=192000.0)
    assert abs(res_cfo) < 150.0
