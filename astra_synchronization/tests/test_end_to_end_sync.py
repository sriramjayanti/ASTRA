"""
test_end_to_end_sync.py
End-to-end integration tests between Stage 5 Candidate Engine and Stage 6 Synchronization Engine.
"""

from astra_synchronization.src.inference import SynchronizationEngine
from astra_synchronization.src.utils import generate_synthetic_test_signal


def test_end_to_end_candidate_sync():
    # Generate synthetic QPSK signal with impairments
    iq, gt = generate_synthetic_test_signal(
        modulation="QPSK",
        symbol_rate_hz=9600.0,
        sample_rate_hz=192000.0,
        cfo_hz=1200.0,
        phase_offset_rad=0.4,
        timing_offset_samples=0.3,
        snr_db=25.0
    )

    engine = SynchronizationEngine()

    # Candidate hypothesis from Stage 5
    hypothesis = {
        "candidate_id": "cand_QPSK_9600_001",
        "modulation": "QPSK",
        "modulation_family": "PSK",
        "symbol_rate_hz": 9600.0,
        "sample_rate_hz": 192000.0,
        "samples_per_symbol": 20.0
    }

    result = engine.synchronize(iq, hypothesis)
    assert result.success is True
    assert result.status in ["SYNC_PASSED", "SYNC_WEAK"]
    assert result.symbol_count > 100
    assert abs(result.estimated_cfo_hz - 1200.0) < 150.0
