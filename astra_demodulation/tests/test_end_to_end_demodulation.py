"""
test_end_to_end_demodulation.py
End-to-end integration test from Stage 6 Synchronization through Stage 7 Demodulation.
"""

from astra_synchronization.src.inference import SynchronizationEngine
from astra_synchronization.src.utils import generate_synthetic_test_signal
from astra_demodulation.src.inference import DemodulationEngine


def test_sync_to_demod_pipeline():
    # 1. Synthesize signal with CFO and timing impairments
    rx_iq, gt = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=1200.0, snr_db=25.0
    )

    # 2. Stage 6 Synchronization
    sync_engine = SynchronizationEngine()
    hyp = {"candidate_id": "cand_e2e_001", "modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0}
    sync_res = sync_engine.synchronize(rx_iq, hyp)
    assert sync_res.success is True

    # 3. Stage 7 Demodulation
    demod_engine = DemodulationEngine()
    demod_res = demod_engine.demodulate(sync_res, hyp)
    assert demod_res.success is True
    assert demod_res.bit_count > 500
    assert len(demod_res.phase_variants) == 4
