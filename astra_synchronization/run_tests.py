"""
run_tests.py
Test runner script for ASTRA Stage 6 — Synchronization Engine.
Runs the 20-test master suite via unittest.
"""

import sys
import os
import unittest

# Ensure workspace and module root are in sys.path
sync_root = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(sync_root)

for p in [workspace_root, sync_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astra_synchronization.tests.test_synchronization import (
    test_1_zero_cfo_unchanged,
    test_2_known_positive_cfo_recovered,
    test_3_known_negative_cfo_recovered,
    test_4_cfo_correction_reduces_residual,
    test_5_rrc_filter_finite_and_symmetric,
    test_6_filter_delay_handled,
    test_7_integer_sps_timing,
    test_8_non_integer_sps_timing,
    test_9_gardner_recovers_known_timing_offset,
    test_10_mueller_muller_works_clean_qpsk,
    test_11_bpsk_costas_locks,
    test_12_qpsk_costas_locks,
    test_13_qam_carrier_recovery_improves_constellation,
    test_14_fsk_route_avoids_psk_carrier_loop,
    test_15_wrong_hypothesis_lower_lock_score,
    test_16_sync_failed_reason_generated,
    test_17_batch_processing,
    test_18_nan_inf_rejected,
    test_19_cpu_execution,
    test_20_json_metadata_serializable,
)


class TestSynchronizationEngine(unittest.TestCase):
    def test_01_zero_cfo_unchanged(self):
        test_1_zero_cfo_unchanged()

    def test_02_known_positive_cfo_recovered(self):
        test_2_known_positive_cfo_recovered()

    def test_03_known_negative_cfo_recovered(self):
        test_3_known_negative_cfo_recovered()

    def test_04_cfo_correction_reduces_residual(self):
        test_4_cfo_correction_reduces_residual()

    def test_05_rrc_filter_finite_and_symmetric(self):
        test_5_rrc_filter_finite_and_symmetric()

    def test_06_filter_delay_handled(self):
        test_6_filter_delay_handled()

    def test_07_integer_sps_timing(self):
        test_7_integer_sps_timing()

    def test_08_non_integer_sps_timing(self):
        test_8_non_integer_sps_timing()

    def test_09_gardner_recovers_known_timing_offset(self):
        test_9_gardner_recovers_known_timing_offset()

    def test_10_mueller_muller_works_clean_qpsk(self):
        test_10_mueller_muller_works_clean_qpsk()

    def test_11_bpsk_costas_locks(self):
        test_11_bpsk_costas_locks()

    def test_12_qpsk_costas_locks(self):
        test_12_qpsk_costas_locks()

    def test_13_qam_carrier_recovery_improves_constellation(self):
        test_13_qam_carrier_recovery_improves_constellation()

    def test_14_fsk_route_avoids_psk_carrier_loop(self):
        test_14_fsk_route_avoids_psk_carrier_loop()

    def test_15_wrong_hypothesis_lower_lock_score(self):
        test_15_wrong_hypothesis_lower_lock_score()

    def test_16_sync_failed_reason_generated(self):
        test_16_sync_failed_reason_generated()

    def test_17_batch_processing(self):
        test_17_batch_processing()

    def test_18_nan_inf_rejected(self):
        test_18_nan_inf_rejected()

    def test_19_cpu_execution(self):
        test_19_cpu_execution()

    def test_20_json_metadata_serializable(self):
        test_20_json_metadata_serializable()


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ASTRA STAGE 6 — SYNCHRONIZATION ENGINE TEST SUITE")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSynchronizationEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("ALL 20 STAGE 6 SYNCHRONIZATION TESTS PASSED (100% SUCCESS)!")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors.")
        sys.exit(1)
