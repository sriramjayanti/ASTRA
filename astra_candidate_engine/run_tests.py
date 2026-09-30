"""
run_tests.py
Test runner script for ASTRA Stage 5 Candidate / Hypothesis Engine.
Supports both unittest and pytest.
"""

import sys
import os
import unittest

# Ensure current directory and project root are in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import tests directly for standard unittest runner
from astra_candidate_engine.tests.test_candidate_engine import (
    test_1_3x3_grid_creates_9_candidates,
    test_2_correct_modulation_rate_pairing,
    test_3_initial_score_calculation,
    test_4_ranking_descending,
    test_5_beam_pruning,
    test_6_duplicate_rates_merged,
    test_7_invalid_rates_rejected,
    test_8_non_integer_sps,
    test_9_rf_evidence_optional,
    test_10_constellation_evidence_optional,
    test_11_unknown_modulation_handling,
    test_12_fallback_rate_mode,
    test_13_source_evidence_preserved,
    test_14_candidate_ids_unique,
    test_15_json_serialization,
    test_16_deterministic_output,
    test_17_max_candidate_bound_respected,
    test_18_rate_variants_controlled,
    test_19_unsupported_modulation_routing,
    test_20_no_nan_inf_scores,
)
from astra_candidate_engine.src.inference import CandidateHypothesisEngine


class TestCandidateHypothesisEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CandidateHypothesisEngine()

    def test_01_3x3_grid_creates_9_candidates(self):
        test_1_3x3_grid_creates_9_candidates(self.engine)

    def test_02_correct_modulation_rate_pairing(self):
        test_2_correct_modulation_rate_pairing(self.engine)

    def test_03_initial_score_calculation(self):
        test_3_initial_score_calculation()

    def test_04_ranking_descending(self):
        test_4_ranking_descending(self.engine)

    def test_05_beam_pruning(self):
        test_5_beam_pruning(self.engine)

    def test_06_duplicate_rates_merged(self):
        test_6_duplicate_rates_merged()

    def test_07_invalid_rates_rejected(self):
        test_7_invalid_rates_rejected(self.engine)

    def test_08_non_integer_sps(self):
        test_8_non_integer_sps(self.engine)

    def test_09_rf_evidence_optional(self):
        test_9_rf_evidence_optional(self.engine)

    def test_10_constellation_evidence_optional(self):
        test_10_constellation_evidence_optional(self.engine)

    def test_11_unknown_modulation_handling(self):
        test_11_unknown_modulation_handling(self.engine)

    def test_12_fallback_rate_mode(self):
        test_12_fallback_rate_mode(self.engine)

    def test_13_source_evidence_preserved(self):
        test_13_source_evidence_preserved(self.engine)

    def test_14_candidate_ids_unique(self):
        test_14_candidate_ids_unique(self.engine)

    def test_15_json_serialization(self):
        test_15_json_serialization(self.engine)

    def test_16_deterministic_output(self):
        test_16_deterministic_output(self.engine)

    def test_17_max_candidate_bound_respected(self):
        test_17_max_candidate_bound_respected()

    def test_18_rate_variants_controlled(self):
        test_18_rate_variants_controlled()

    def test_19_unsupported_modulation_routing(self):
        test_19_unsupported_modulation_routing(self.engine)

    def test_20_no_nan_inf_scores(self):
        test_20_no_nan_inf_scores(self.engine)


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ASTRA STAGE 5 — CANDIDATE / HYPOTHESIS ENGINE TEST SUITE")
    print("=" * 70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCandidateHypothesisEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("ALL 20 STAGE 5 CANDIDATE ENGINE TESTS PASSED (100% SUCCESS)!")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors.")
        sys.exit(1)
