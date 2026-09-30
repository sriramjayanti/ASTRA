"""
test_mueller_muller.py
Unit tests for Mueller & Müller Decision-Directed Timing Recovery.
"""

from astra_synchronization.src.mueller_muller import mueller_muller_recover
from astra_synchronization.src.gardner import gardner_recover
from astra_synchronization.src.utils import generate_synthetic_test_signal


def test_mueller_muller_refinement():
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=19200.0, snr_db=30.0
    )
    syms, _, _, _ = gardner_recover(iq, sps=2.0)
    ref_syms, t_err, mm_lock, state = mueller_muller_recover(syms, modulation="QPSK")
    assert len(ref_syms) == len(syms) - 1
    assert mm_lock > 0.4
