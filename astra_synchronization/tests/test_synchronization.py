"""
test_synchronization.py
Master test suite for ASTRA Stage 6 Synchronization Engine covering all 20 mandatory unit tests.
"""

import math
import json
import numpy as np

from astra_synchronization.src.inference import SynchronizationEngine
from astra_synchronization.src.models import SynchronizationResult, SyncStatus
from astra_synchronization.src.cfo import (
    estimate_cfo,
    correct_cfo,
    estimate_coarse_cfo_mth_power,
    estimate_residual_cfo
)
from astra_synchronization.src.matched_filter import design_rrc_filter, apply_matched_filter
from astra_synchronization.src.resampling import resample_to_working_sps
from astra_synchronization.src.gardner import gardner_recover
from astra_synchronization.src.mueller_muller import mueller_muller_recover
from astra_synchronization.src.costas import costas_recover
from astra_synchronization.src.carrier import qam_carrier_recover
from astra_synchronization.src.fsk_sync import synchronize_fsk_pipeline
from astra_synchronization.src.quality import calculate_constellation_compactness
from astra_synchronization.src.utils import generate_synthetic_test_signal


def get_engine():
    return SynchronizationEngine()


# TEST 1: zero CFO unchanged
def test_1_zero_cfo_unchanged():
    iq, gt = generate_synthetic_test_signal(
        modulation="QPSK", cfo_hz=0.0, phase_offset_rad=0.0, snr_db=30.0
    )
    cfo_est, _ = estimate_cfo(iq, sample_rate_hz=192000.0, modulation="QPSK", modulation_family="PSK")
    assert abs(cfo_est) < 50.0  # Within 50 Hz at 192 kHz Fs (0.025% of Fs)


# TEST 2: known positive CFO recovered
def test_2_known_positive_cfo_recovered():
    true_cfo = 1500.0
    iq, gt = generate_synthetic_test_signal(
        modulation="QPSK", cfo_hz=true_cfo, snr_db=30.0
    )
    cfo_est, _ = estimate_cfo(iq, sample_rate_hz=192000.0, modulation="QPSK", modulation_family="PSK")
    assert abs(cfo_est - true_cfo) < 100.0


# TEST 3: known negative CFO recovered
def test_3_known_negative_cfo_recovered():
    true_cfo = -2200.0
    iq, gt = generate_synthetic_test_signal(
        modulation="QPSK", cfo_hz=true_cfo, snr_db=30.0
    )
    cfo_est, _ = estimate_cfo(iq, sample_rate_hz=192000.0, modulation="QPSK", modulation_family="PSK")
    assert abs(cfo_est - true_cfo) < 100.0


# TEST 4: CFO correction reduces residual
def test_4_cfo_correction_reduces_residual():
    true_cfo = 1800.0
    iq, gt = generate_synthetic_test_signal(
        modulation="QPSK", cfo_hz=true_cfo, snr_db=30.0
    )
    cfo_est, _ = estimate_cfo(iq, sample_rate_hz=192000.0, modulation="QPSK", modulation_family="PSK")
    corrected = correct_cfo(iq, cfo_hz=cfo_est, sample_rate_hz=192000.0)
    
    # Residual CFO after correction should be very close to 0
    res_cfo = estimate_residual_cfo(corrected, sample_rate_hz=192000.0)
    assert abs(res_cfo) < 150.0


# TEST 5: RRC filter finite/symmetric
def test_5_rrc_filter_finite_and_symmetric():
    h = design_rrc_filter(sps=20.0, rolloff=0.35, span_symbols=8)
    assert len(h) > 0
    assert np.all(np.isfinite(h))
    # Check symmetric FIR response
    assert np.allclose(h, h[::-1], atol=1e-5)


# TEST 6: filter delay handled
def test_6_filter_delay_handled():
    iq, _ = generate_synthetic_test_signal(modulation="QPSK", snr_db=30.0)
    filtered, delay = apply_matched_filter(iq, sps=20.0, rolloff=0.35, span_symbols=8)
    assert delay > 0
    assert len(filtered) == len(iq)


# TEST 7: integer SPS timing
def test_7_integer_sps_timing():
    # 192000 / 9600 = 20.0 integer SPS
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=0.0, snr_db=30.0
    )
    engine = get_engine()
    res = engine.synchronize(iq, {"modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    assert res.success is True
    assert res.symbol_count > 100


# TEST 8: non-integer SPS timing
def test_8_non_integer_sps_timing():
    # 192000 / 7000 = 27.42857 non-integer SPS
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=7000.0, sample_rate_hz=192000.0, cfo_hz=0.0, snr_db=30.0
    )
    engine = get_engine()
    res = engine.synchronize(iq, {"modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 7000.0, "sample_rate_hz": 192000.0})
    assert res.success is True
    assert res.symbol_count > 50


# TEST 9: Gardner recovers known timing offset
def test_9_gardner_recovers_known_timing_offset():
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=19200.0, cfo_hz=0.0, timing_offset_samples=0.4, snr_db=30.0
    )
    # SPS = 2.0
    symbols, t_offset, t_lock, state = gardner_recover(iq, sps=2.0)
    assert len(symbols) > 50
    assert t_lock > 0.5


# TEST 10: M&M works on clean QPSK
def test_10_mueller_muller_works_clean_qpsk():
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=19200.0, cfo_hz=0.0, snr_db=30.0
    )
    symbols, _, _, _ = gardner_recover(iq, sps=2.0)
    refined_syms, t_err, mm_lock, _ = mueller_muller_recover(symbols, modulation="QPSK")
    assert len(refined_syms) == len(symbols) - 1
    assert mm_lock > 0.5


# TEST 11: BPSK Costas locks
def test_11_bpsk_costas_locks():
    iq, _ = generate_synthetic_test_signal(
        modulation="BPSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=500.0, phase_offset_rad=0.6, snr_db=25.0
    )
    engine = get_engine()
    res = engine.synchronize(iq, {"modulation": "BPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    assert res.success is True
    assert res.lock_metrics["carrier_lock_score"] > 0.45


# TEST 12: QPSK Costas locks
def test_12_qpsk_costas_locks():
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=800.0, phase_offset_rad=0.4, snr_db=25.0
    )
    engine = get_engine()
    res = engine.synchronize(iq, {"modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    assert res.success is True
    assert res.lock_metrics["carrier_lock_score"] > 0.45


# TEST 13: QAM carrier recovery improves constellation
def test_13_qam_carrier_recovery_improves_constellation():
    iq, _ = generate_synthetic_test_signal(
        modulation="16-QAM", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=400.0, snr_db=25.0
    )
    engine = get_engine()
    res = engine.synchronize(iq, {"modulation": "16-QAM", "modulation_family": "QAM", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    assert res.symbol_count > 50
    assert res.lock_metrics["overall_sync_score"] > 0.40


# TEST 14: FSK route avoids PSK carrier loop
def test_14_fsk_route_avoids_psk_carrier_loop():
    iq, _ = generate_synthetic_test_signal(
        modulation="2-FSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=500.0, snr_db=25.0
    )
    engine = get_engine()
    res = engine.synchronize(iq, {"modulation": "2-FSK", "modulation_family": "FSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    assert res.carrier_method == "tone_centering"
    assert res.matched_filter_type == "none_discriminator"
    assert res.symbol_count > 50


# TEST 15: wrong hypothesis often yields lower lock score
def test_15_wrong_hypothesis_lower_lock_score():
    # True signal: QPSK @ 9600 Bd
    iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=600.0, snr_db=25.0
    )
    engine = get_engine()
    
    # Correct hypothesis
    res_correct = engine.synchronize(iq, {"candidate_id": "cand_correct", "modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    
    # Wrong rate hypothesis (e.g. 4800 Bd instead of 9600 Bd)
    res_wrong = engine.synchronize(iq, {"candidate_id": "cand_wrong", "modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 4800.0, "sample_rate_hz": 192000.0})
    
    score_correct = res_correct.lock_metrics.get("overall_sync_score", 0.0)
    score_wrong = res_wrong.lock_metrics.get("overall_sync_score", 0.0)
    assert score_correct > score_wrong


# TEST 16: SYNC_FAILED reason generated
def test_16_sync_failed_reason_generated():
    # Unsupported / unknown family
    engine = get_engine()
    iq = np.zeros(500, dtype=np.complex64)
    res = engine.synchronize(iq, {"candidate_id": "cand_fail", "modulation": "UNKNOWN", "modulation_family": "UNKNOWN", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    assert res.status == SyncStatus.SYNC_FAILED.value
    assert res.failure_reason is not None


# TEST 17: batch processing
def test_17_batch_processing():
    iq, _ = generate_synthetic_test_signal(modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0)
    engine = get_engine()
    hypotheses = [
        {"candidate_id": "cand_1", "modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0},
        {"candidate_id": "cand_2", "modulation": "8PSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0},
    ]
    batch_results = engine.synchronize_batch(iq, hypotheses, sample_rate_hz=192000.0)
    assert len(batch_results) == 2
    assert batch_results[0].candidate_id == "cand_1"
    assert batch_results[1].candidate_id == "cand_2"


# TEST 18: NaN/Inf rejected
def test_18_nan_inf_rejected():
    engine = get_engine()
    iq_nan = np.array([1.0 + 1j * 0.0, float('nan') + 1j * 0.0, 0.5 + 1j * 0.5] * 50)
    res = engine.synchronize(iq_nan, {"modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    assert res.status == SyncStatus.SYNC_FAILED.value
    assert "non-finite" in res.failure_reason


# TEST 19: CPU execution
def test_19_cpu_execution():
    # Verify standard pure CPU execution using NumPy/SciPy
    iq, _ = generate_synthetic_test_signal(modulation="QPSK", num_symbols=200, snr_db=30.0)
    engine = get_engine()
    res = engine.synchronize(iq, {"modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    assert isinstance(res.symbol_samples, np.ndarray)


# TEST 20: JSON metadata serializable
def test_20_json_metadata_serializable():
    iq, _ = generate_synthetic_test_signal(modulation="QPSK", snr_db=30.0)
    engine = get_engine()
    res = engine.synchronize(iq, {"candidate_id": "cand_json", "modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    json_str = res.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["candidate_id"] == "cand_json"
    assert "lock_metrics" in parsed
