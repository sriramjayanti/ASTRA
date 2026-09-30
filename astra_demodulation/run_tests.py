"""
run_tests.py
Test runner script for ASTRA Stage 7 — Demodulation Engine.
Runs the complete 30-test master suite via unittest.
"""

import sys
import os
import unittest

demod_root = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(demod_root)

for p in [workspace_root, demod_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astra_demodulation.tests.test_demodulation import (
    test_1_bpsk_ideal_ber_zero,
    test_2_qpsk_ideal_ber_zero,
    test_3_8psk_ideal_ber_zero,
    test_4_16qam_ideal_ber_zero,
    test_5_64qam_ideal_ber_zero,
    test_6_2fsk_ideal_ber_zero,
    test_7_4fsk_ideal_ber_zero,
    test_8_bpsk_llr_sign_correct,
    test_9_qpsk_llr_bits_correct,
    test_10_16qam_llr_bits_correct,
    test_11_64qam_llr_bits_correct,
    test_12_max_log_vs_exact_llr,
    test_13_qpsk_90_deg_ambiguity,
    test_14_qpsk_180_deg_ambiguity,
    test_15_qpsk_270_deg_ambiguity,
    test_16_8psk_ambiguity_supported,
    test_17_qam_rotations_supported,
    test_18_evm_near_zero_perfect_constellation,
    test_19_evm_worsens_with_noise,
    test_20_llr_magnitude_falls_with_noise,
    test_21_fsk_tone_energies_correct,
    test_22_hard_bits_and_llr_ordering_identical,
    test_23_nan_input_rejected,
    test_24_unsupported_modulation_rejected,
    test_25_batch_inference,
    test_26_large_input_chunking,
    test_27_cpu_only_execution,
    test_28_json_metadata_serialization,
    test_29_stage_6_result_integration,
    test_30_deterministic_ideal_output,
)


class TestDemodulationEngine(unittest.TestCase):
    def test_01_bpsk_ideal_ber_zero(self):
        test_1_bpsk_ideal_ber_zero()

    def test_02_qpsk_ideal_ber_zero(self):
        test_2_qpsk_ideal_ber_zero()

    def test_03_8psk_ideal_ber_zero(self):
        test_3_8psk_ideal_ber_zero()

    def test_04_16qam_ideal_ber_zero(self):
        test_4_16qam_ideal_ber_zero()

    def test_05_64qam_ideal_ber_zero(self):
        test_5_64qam_ideal_ber_zero()

    def test_06_2fsk_ideal_ber_zero(self):
        test_6_2fsk_ideal_ber_zero()

    def test_07_4fsk_ideal_ber_zero(self):
        test_7_4fsk_ideal_ber_zero()

    def test_08_bpsk_llr_sign_correct(self):
        test_8_bpsk_llr_sign_correct()

    def test_09_qpsk_llr_bits_correct(self):
        test_9_qpsk_llr_bits_correct()

    def test_10_16qam_llr_bits_correct(self):
        test_10_16qam_llr_bits_correct()

    def test_11_64qam_llr_bits_correct(self):
        test_11_64qam_llr_bits_correct()

    def test_12_max_log_vs_exact_llr(self):
        test_12_max_log_vs_exact_llr()

    def test_13_qpsk_90_deg_ambiguity(self):
        test_13_qpsk_90_deg_ambiguity()

    def test_14_qpsk_180_deg_ambiguity(self):
        test_14_qpsk_180_deg_ambiguity()

    def test_15_qpsk_270_deg_ambiguity(self):
        test_15_qpsk_270_deg_ambiguity()

    def test_16_8psk_ambiguity_supported(self):
        test_16_8psk_ambiguity_supported()

    def test_17_qam_rotations_supported(self):
        test_17_qam_rotations_supported()

    def test_18_evm_near_zero_perfect_constellation(self):
        test_18_evm_near_zero_perfect_constellation()

    def test_19_evm_worsens_with_noise(self):
        test_19_evm_worsens_with_noise()

    def test_20_llr_magnitude_falls_with_noise(self):
        test_20_llr_magnitude_falls_with_noise()

    def test_21_fsk_tone_energies_correct(self):
        test_21_fsk_tone_energies_correct()

    def test_22_hard_bits_and_llr_ordering_identical(self):
        test_22_hard_bits_and_llr_ordering_identical()

    def test_23_nan_input_rejected(self):
        test_23_nan_input_rejected()

    def test_24_unsupported_modulation_rejected(self):
        test_24_unsupported_modulation_rejected()

    def test_25_batch_inference(self):
        test_25_batch_inference()

    def test_26_large_input_chunking(self):
        test_26_large_input_chunking()

    def test_27_cpu_only_execution(self):
        test_27_cpu_only_execution()

    def test_28_json_metadata_serialization(self):
        test_28_json_metadata_serialization()

    def test_29_stage_6_result_integration(self):
        test_29_stage_6_result_integration()

    def test_30_deterministic_ideal_output(self):
        test_30_deterministic_ideal_output()


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ASTRA STAGE 7 — DEMODULATION ENGINE TEST SUITE")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDemodulationEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("ALL 30 STAGE 7 DEMODULATION TESTS PASSED (100% SUCCESS)!")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors.")
        sys.exit(1)
