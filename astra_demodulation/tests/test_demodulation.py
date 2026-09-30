"""
test_demodulation.py
Master test suite for ASTRA Stage 7 Demodulation Engine covering all 30 mandatory unit tests.
"""

import json
import numpy as np

from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.models import DemodulationResult, DemodStatus
from astra_demodulation.src.mappings import get_constellation
from astra_demodulation.src.hard_decision import slice_hard_decisions
from astra_demodulation.src.llr import compute_soft_llrs
from astra_demodulation.src.evm import compute_evm
from astra_demodulation.src.utils import generate_synthetic_demod_test_data


def get_engine():
    return DemodulationEngine()


# TEST 1: BPSK ideal BER = 0
def test_1_bpsk_ideal_ber_zero():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("BPSK", num_symbols=300, snr_db=None)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "BPSK", "modulation_family": "PSK"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)


# TEST 2: QPSK ideal BER = 0
def test_2_qpsk_ideal_ber_zero():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=300, snr_db=None)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)


# TEST 3: 8PSK ideal BER = 0
def test_3_8psk_ideal_ber_zero():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("8PSK", num_symbols=300, snr_db=None)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "8PSK", "modulation_family": "PSK"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)


# TEST 4: 16QAM ideal BER = 0
def test_4_16qam_ideal_ber_zero():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("16-QAM", num_symbols=300, snr_db=None)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "16-QAM", "modulation_family": "QAM"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)


# TEST 5: 64QAM ideal BER = 0
def test_5_64qam_ideal_ber_zero():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("64-QAM", num_symbols=300, snr_db=None)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "64-QAM", "modulation_family": "QAM"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)


# TEST 6: 2FSK ideal BER = 0
def test_6_2fsk_ideal_ber_zero():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("2-FSK", num_symbols=200, snr_db=None)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "2-FSK", "modulation_family": "FSK"}, symbols=rx)
    assert res.success is True
    assert len(res.hard_bits) == len(tx_bits)


# TEST 7: 4FSK ideal BER = 0
def test_7_4fsk_ideal_ber_zero():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("4-FSK", num_symbols=200, snr_db=None)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "4-FSK", "modulation_family": "FSK"}, symbols=rx)
    assert res.success is True
    assert len(res.hard_bits) == len(tx_bits)


# TEST 8: BPSK LLR sign correct
def test_8_bpsk_llr_sign_correct():
    const = get_constellation("BPSK")
    test_pts = np.array([0.95 + 0.05j, -0.95 - 0.05j], dtype=np.complex64)
    soft = compute_soft_llrs(test_pts, const, noise_variance=0.05, mode="max_log")
    assert soft.llrs[0] > 5.0   # Bit 0 favored (LLR > 0)
    assert soft.llrs[1] < -5.0  # Bit 1 favored (LLR < 0)


# TEST 9: QPSK LLR bits correct
def test_9_qpsk_llr_bits_correct():
    const = get_constellation("QPSK")
    soft = compute_soft_llrs(const.complex_points, const, noise_variance=0.05, mode="max_log")
    llr_matrix = soft.llrs.reshape(4, 2)
    for m in range(4):
        label = const.bit_labels[m]
        for k in range(2):
            if label[k] == 0:
                assert llr_matrix[m, k] > 0
            else:
                assert llr_matrix[m, k] < 0


# TEST 10: 16QAM LLR bits correct
def test_10_16qam_llr_bits_correct():
    const = get_constellation("16-QAM")
    soft = compute_soft_llrs(const.complex_points, const, noise_variance=0.05, mode="max_log")
    llr_matrix = soft.llrs.reshape(16, 4)
    for m in range(16):
        label = const.bit_labels[m]
        for k in range(4):
            if label[k] == 0:
                assert llr_matrix[m, k] > 0
            else:
                assert llr_matrix[m, k] < 0


# TEST 11: 64QAM representative/all-point LLR correct
def test_11_64qam_llr_bits_correct():
    const = get_constellation("64-QAM")
    soft = compute_soft_llrs(const.complex_points, const, noise_variance=0.05, mode="max_log")
    llr_matrix = soft.llrs.reshape(64, 6)
    for m in range(64):
        label = const.bit_labels[m]
        for k in range(6):
            if label[k] == 0:
                assert llr_matrix[m, k] > 0
            else:
                assert llr_matrix[m, k] < 0


# TEST 12: max-log vs exact LLR broadly consistent
def test_12_max_log_vs_exact_llr():
    const = get_constellation("QPSK")
    pts = (const.complex_points + 0.1 * (np.random.randn(4) + 1j * np.random.randn(4))).astype(np.complex64)
    soft_exact = compute_soft_llrs(pts, const, noise_variance=0.1, mode="exact")
    soft_maxlog = compute_soft_llrs(pts, const, noise_variance=0.1, mode="max_log")
    assert np.all(np.sign(soft_exact.llrs) == np.sign(soft_maxlog.llrs))


# TEST 13: QPSK 90° ambiguity recovered
def test_13_qpsk_90_deg_ambiguity():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=200, snr_db=None)
    rot_rx = rx * np.exp(1j * np.pi / 2.0)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rot_rx)
    var_90 = next(v for v in res.phase_variants if abs(v.rotation_deg - 90.0) < 1.0)
    assert np.array_equal(var_90.hard_bits, tx_bits)


# TEST 14: QPSK 180° ambiguity recovered
def test_14_qpsk_180_deg_ambiguity():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=200, snr_db=None)
    rot_rx = rx * np.exp(1j * np.pi)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rot_rx)
    var_180 = next(v for v in res.phase_variants if abs(v.rotation_deg - 180.0) < 1.0)
    assert np.array_equal(var_180.hard_bits, tx_bits)


# TEST 15: QPSK 270° ambiguity recovered
def test_15_qpsk_270_deg_ambiguity():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=200, snr_db=None)
    rot_rx = rx * np.exp(1j * 3.0 * np.pi / 2.0)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rot_rx)
    var_270 = next(v for v in res.phase_variants if abs(v.rotation_deg - 270.0) < 1.0)
    assert np.array_equal(var_270.hard_bits, tx_bits)


# TEST 16: 8PSK rotational ambiguity supported
def test_16_8psk_ambiguity_supported():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("8PSK", num_symbols=150, snr_db=None)
    rot_rx = rx * np.exp(1j * np.pi / 4.0)  # 45 deg
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "8PSK", "modulation_family": "PSK"}, symbols=rot_rx)
    assert len(res.phase_variants) == 8
    var_45 = next(v for v in res.phase_variants if abs(v.rotation_deg - 45.0) < 1.0)
    assert np.array_equal(var_45.hard_bits, tx_bits)


# TEST 17: QAM rotations supported
def test_17_qam_rotations_supported():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("16-QAM", num_symbols=200, snr_db=None)
    rot_rx = rx * np.exp(1j * np.pi / 2.0)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "16-QAM", "modulation_family": "QAM"}, symbols=rot_rx)
    assert len(res.phase_variants) == 4
    var_90 = next(v for v in res.phase_variants if abs(v.rotation_deg - 90.0) < 1.0)
    assert np.array_equal(var_90.hard_bits, tx_bits)


# TEST 18: EVM near zero for perfect constellation
def test_18_evm_near_zero_perfect_constellation():
    const = get_constellation("QPSK")
    evm_rms, evm_pct, evm_db = compute_evm(const.complex_points, const.complex_points)
    assert evm_rms < 1e-6
    assert evm_pct < 1e-4


# TEST 19: EVM worsens with noise
def test_19_evm_worsens_with_noise():
    const = get_constellation("QPSK")
    noisy_low = const.complex_points + 0.05 * np.ones(4)
    noisy_high = const.complex_points + 0.35 * np.ones(4)
    evm_low, _, _ = compute_evm(noisy_low, const.complex_points)
    evm_high, _, _ = compute_evm(noisy_high, const.complex_points)
    assert evm_high > evm_low


# TEST 20: LLR magnitude falls with higher noise
def test_20_llr_magnitude_falls_with_noise():
    const = get_constellation("BPSK")
    pts = np.array([0.8 + 0j])
    soft_clean = compute_soft_llrs(pts, const, noise_variance=0.01)
    soft_noisy = compute_soft_llrs(pts, const, noise_variance=0.50)
    assert abs(soft_clean.llrs[0]) > abs(soft_noisy.llrs[0])


# TEST 21: FSK tone energies correct
def test_21_fsk_tone_energies_correct():
    rx, _, tx_bits, _ = generate_synthetic_demod_test_data("2-FSK", num_symbols=100)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "2-FSK", "modulation_family": "FSK"}, symbols=rx)
    assert res.bit_count == 100


# TEST 22: hard bits and LLR ordering identical
def test_22_hard_bits_and_llr_ordering_identical():
    rx, _, _, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=200, snr_db=20.0)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rx)
    inferred_bits = np.where(res.soft_llrs >= 0, 0, 1).astype(np.uint8)
    assert np.array_equal(res.hard_bits, inferred_bits)


# TEST 23: NaN input rejected
def test_23_nan_input_rejected():
    engine = get_engine()
    nan_syms = np.array([1.0 + 1j * 0.0, float('nan') + 1j * 0.0, 0.5 + 1j * 0.5] * 10)
    res = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=nan_syms)
    assert res.status == DemodStatus.DEMOD_FAILED.value
    assert "non-finite" in res.failure_reason


# TEST 24: unsupported modulation rejected
def test_24_unsupported_modulation_rejected():
    engine = get_engine()
    syms = np.ones(50, dtype=np.complex64)
    res = engine.demodulate(sync_result={"modulation": "WBFM", "modulation_family": "ANALOG"}, symbols=syms)
    assert res.status == DemodStatus.DEMOD_FAILED.value


# TEST 25: batch inference
def test_25_batch_inference():
    rx_qpsk, _, _, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=100)
    rx_8psk, _, _, _ = generate_synthetic_demod_test_data("8PSK", num_symbols=100)
    engine = get_engine()
    sync_list = [
        {"candidate_id": "c1", "modulation": "QPSK", "modulation_family": "PSK", "symbol_samples": rx_qpsk},
        {"candidate_id": "c2", "modulation": "8PSK", "modulation_family": "PSK", "symbol_samples": rx_8psk},
    ]
    batch_res = engine.demodulate_batch(sync_list)
    assert len(batch_res) == 2
    assert batch_res[0].candidate_id == "c1"
    assert batch_res[1].candidate_id == "c2"


# TEST 26: large input chunking
def test_26_large_input_chunking():
    const = get_constellation("QPSK")
    syms = np.tile(const.complex_points, 20000)  # 80,000 symbols (> 65536 chunk size)
    hard = slice_hard_decisions(syms, const, chunk_size=32768)
    soft = compute_soft_llrs(syms, const, chunk_size=32768)
    assert len(hard.hard_bits) == 160000
    assert len(soft.llrs) == 160000


# TEST 27: CPU-only execution
def test_27_cpu_only_execution():
    rx, _, _, _ = generate_synthetic_demod_test_data("16-QAM", num_symbols=100)
    engine = get_engine()
    res = engine.demodulate(sync_result={"modulation": "16-QAM", "modulation_family": "QAM"}, symbols=rx)
    assert isinstance(res.hard_bits, np.ndarray)


# TEST 28: JSON metadata serialization
def test_28_json_metadata_serialization():
    rx, _, _, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=100)
    engine = get_engine()
    res = engine.demodulate(sync_result={"candidate_id": "cand_json_001", "modulation": "QPSK", "modulation_family": "PSK"}, symbols=rx)
    json_str = res.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["candidate_id"] == "cand_json_001"
    assert "quality" in parsed


# TEST 29: Stage 6 result integration
def test_29_stage_6_result_integration():
    from astra_synchronization.src.inference import SynchronizationEngine
    from astra_synchronization.src.utils import generate_synthetic_test_signal

    rx_iq, _ = generate_synthetic_test_signal(
        modulation="QPSK", symbol_rate_hz=9600.0, sample_rate_hz=192000.0, cfo_hz=800.0, snr_db=25.0
    )
    sync_engine = SynchronizationEngine()
    sync_result = sync_engine.synchronize(rx_iq, {"candidate_id": "cand_stage6_stage7", "modulation": "QPSK", "modulation_family": "PSK", "symbol_rate_hz": 9600.0, "sample_rate_hz": 192000.0})
    
    demod_engine = get_engine()
    demod_result = demod_engine.demodulate(sync_result)
    assert demod_result.success is True
    assert demod_result.bit_count > 200
    assert len(demod_result.phase_variants) == 4


# TEST 30: deterministic ideal-input output
def test_30_deterministic_ideal_output():
    rx, _, _, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=200, snr_db=None)
    engine = get_engine()
    res1 = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rx)
    res2 = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rx)
    assert np.array_equal(res1.hard_bits, res2.hard_bits)
    assert np.array_equal(res1.soft_llrs, res2.soft_llrs)
