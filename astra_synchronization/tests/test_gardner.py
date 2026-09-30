"""
test_gardner.py
Unit tests for Gardner Timing Error Detector.
"""

from astra_synchronization.src.gardner import gardner_recover
from astra_synchronization.src.utils import generate_synthetic_test_signal


def test_gardner_timing():
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=19200.0, snr_db=30.0
    )
    symbols, t_offset, lock_score, state = gardner_recover(iq, sps=2.0)
    assert len(symbols) > 50
    assert lock_score > 0.4
