"""
test_scoring.py
Unit tests for structural feature extraction and deterministic rule-based candidate scoring.
"""

import unittest
import numpy as np
from astra_interleaver.src.structural_features import (
    extract_structural_features,
    compute_binary_entropy,
    compute_run_length_statistics,
    compute_autocorrelation_features
)
from astra_interleaver.src.scoring import CandidateScorer
from astra_interleaver.src.models import InterleaverStatus


class TestStructuralScoring(unittest.TestCase):
    def test_entropy_calculation(self):
        # All zeros -> 0 entropy
        zeros = np.zeros(200, dtype=np.uint8)
        self.assertAlmostEqual(compute_binary_entropy(zeros), 0.0, places=4)
        
        # Perfect 50/50 balance -> 1.0 entropy
        balanced = np.array([0, 1] * 100, dtype=np.uint8)
        self.assertAlmostEqual(compute_binary_entropy(balanced), 1.0, places=4)

    def test_run_length_statistics(self):
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1], dtype=np.uint8)
        mean_run, max_run, rl_entropy = compute_run_length_statistics(bits)
        self.assertEqual(max_run, 4)
        self.assertGreater(mean_run, 0.0)

    def test_autocorrelation_periodicity(self):
        # Highly periodic sequence with period 16
        pattern = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        periodic_stream = np.tile(pattern, 32)
        peak, lag, periodicity = compute_autocorrelation_features(periodic_stream)
        self.assertGreater(periodicity, 0.5)

    def test_candidate_scorer_deterministic(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        feat = extract_structural_features(bits)
        scorer = CandidateScorer()
        
        s1, _, _, _, _, st1 = scorer.score_candidate(feat, "block")
        s2, _, _, _, _, st2 = scorer.score_candidate(feat, "block")
        self.assertEqual(s1, s2)
        self.assertEqual(st1, st2)


if __name__ == "__main__":
    unittest.main()
