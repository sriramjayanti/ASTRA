"""
ASTRA Constellation Analysis Engine Unit Test Suite.
"""

import sys
import unittest
import numpy as np

from astra_constellation.src.clustering import (
    evaluate_dbscan_clustering,
    evaluate_kmeans_clustering,
    extract_iq_coordinates,
    extract_radial_rings,
)
from astra_constellation.src.geometry import (
    compute_angular_uniformity,
    compute_evm,
    compute_square_grid_compactness,
    get_reference_constellation,
)
from astra_constellation.src.inference import ConstellationAnalyzer
from astra_constellation.src.models import InvalidIQError, ConstellationPrediction


class TestConstellationEngine(unittest.TestCase):

    def setUp(self):
        self.analyzer = ConstellationAnalyzer()

    def test_01_extract_coordinates_shape_and_norm(self):
        """Test conversion of complex array to normalized real [N, 2] coordinates."""
        iq = np.random.normal(0, 1, 500) + 1j * np.random.normal(0, 1, 500)
        coords = extract_iq_coordinates(iq)
        self.assertEqual(coords.shape, (500, 2))
        self.assertAlmostEqual(float(np.mean(coords[:, 0]**2 + coords[:, 1]**2)), 1.0, places=3)

    def test_02_bpsk_clustering(self):
        """Test spatial clustering on pure BPSK symbols (2 clusters)."""
        bits = np.random.choice([-1.0, 1.0], size=1000)
        noise = (np.random.normal(0, 0.05, 1000) + 1j * np.random.normal(0, 0.05, 1000))
        bpsk_iq = (bits + noise).astype(np.complex64)

        pred = self.analyzer.analyze(bpsk_iq)
        self.assertEqual(pred.best_constellation, "BPSK")
        self.assertEqual(pred.m_ary_order, 2)
        self.assertIn(pred.status, ["CONFIRMED", "ESTIMATED"])

    def test_03_qpsk_clustering(self):
        """Test spatial clustering on pure QPSK symbols (4 clusters)."""
        grid = np.array([-1.0, 1.0])
        i_syms = np.random.choice(grid, 1000)
        q_syms = np.random.choice(grid, 1000)
        noise = (np.random.normal(0, 0.05, 1000) + 1j * np.random.normal(0, 0.05, 1000))
        qpsk_iq = ((i_syms + 1j * q_syms) / np.sqrt(2.0) + noise).astype(np.complex64)

        pred = self.analyzer.analyze(qpsk_iq)
        self.assertEqual(pred.best_constellation, "QPSK")
        self.assertEqual(pred.m_ary_order, 4)
        self.assertIn(pred.status, ["CONFIRMED", "ESTIMATED"])

    def test_04_16qam_radial_rings(self):
        """Test 16-QAM radial ring identification (3 distinct amplitude levels)."""
        grid = np.array([-3.0, -1.0, 1.0, 3.0])
        i_syms = np.random.choice(grid, 2000)
        q_syms = np.random.choice(grid, 2000)
        noise = (np.random.normal(0, 0.03, 2000) + 1j * np.random.normal(0, 0.03, 2000))
        qam_iq = ((i_syms + 1j * q_syms) / np.sqrt(10.0) + noise).astype(np.complex64)

        ring_count, ring_amps, cm_var = extract_radial_rings(qam_iq)
        self.assertEqual(ring_count, 3)
        self.assertTrue(cm_var > 0.05)  # Multi-amplitude signal

    def test_05_evm_zero_for_ideal_template(self):
        """Test EVM calculation gives ~0% for ideal points."""
        ref_qpsk = get_reference_constellation("QPSK")
        evm = compute_evm(ref_qpsk, ref_qpsk)
        self.assertAlmostEqual(evm, 0.0, places=2)

    def test_06_angular_uniformity_for_psk(self):
        """Test phase uniformity metric is high for clean QPSK."""
        ref_qpsk = get_reference_constellation("QPSK")
        repeated = np.tile(ref_qpsk, 200)
        score = compute_angular_uniformity(repeated, m_ary=4)
        self.assertTrue(score > 0.85)

    def test_07_nan_and_empty_handling(self):
        """Test invalid IQ handling."""
        with self.assertRaises(InvalidIQError):
            self.analyzer.analyze(np.array([1.0, np.nan, 2.0]))
        with self.assertRaises(InvalidIQError):
            self.analyzer.analyze(np.array([]))

    def test_08_pure_noise_behavior(self):
        """Test pure noise produces UNKNOWN or low confidence."""
        noise = (np.random.normal(0, 1, 1000) + 1j * np.random.normal(0, 1, 1000)).astype(np.complex64)
        pred = self.analyzer.analyze(noise)
        self.assertIsInstance(pred, ConstellationPrediction)
        self.assertTrue(pred.confidence < 0.65)


if __name__ == "__main__":
    unittest.main()
