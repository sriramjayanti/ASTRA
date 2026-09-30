"""
test_fsk_sync.py
Unit tests for FSK synchronization pipeline.
"""

from astra_synchronization.src.fsk_sync import synchronize_fsk_pipeline
from astra_synchronization.src.utils import generate_synthetic_test_signal


def test_fsk_pipeline():
    iq, _ = generate_synthetic_test_signal(
        modulation="2-FSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=600.0, snr_db=25.0
    )
    corr_iq, syms, cfo_est, t_lock, f_lock, meta = synchronize_fsk_pipeline(
        iq, sample_rate_hz=192000.0, symbol_rate_hz=9600.0, modulation="2-FSK"
    )
    assert len(syms) > 50
    assert t_lock > 0.3
    assert f_lock > 0.4
