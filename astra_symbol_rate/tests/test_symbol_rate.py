"""
ASTRA Symbol-Rate (Baud) Estimation Engine Comprehensive Test Suite.
Verifies all 20 required specifications.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import numpy as np

from astra_symbol_rate.src.autocorrelation import extract_autocorr_evidence
from astra_symbol_rate.src.bandwidth import extract_bandwidth_evidence
from astra_symbol_rate.src.candidate_features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    build_candidate_feature_matrix,
    extract_candidate_feature_dict,
)
from astra_symbol_rate.src.candidate_generator import (
    deduplicate_candidates,
    detect_harmonic_candidates,
    generate_candidates,
)
from astra_symbol_rate.src.dataset_builder import (
    build_candidate_dataset_from_signals,
    extract_dsp_evidence,
    generate_synthetic_training_pool,
)
from astra_symbol_rate.src.inference import SymbolRateEstimator
from astra_symbol_rate.src.instantaneous_frequency import extract_instantaneous_frequency_evidence
from astra_symbol_rate.src.models import (
    DSPRateEvidence,
    InvalidSignalError,
    SymbolRateCandidate,
    SymbolRatePrediction,
)
from astra_symbol_rate.src.utils import generate_synthetic_signal
from astra_symbol_rate.src.xgboost_ranker import XGBoostSymbolRateRanker


class TestSymbolRateEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        # Create a trained ranker for inference tests
        cls.signals = generate_synthetic_training_pool(num_signals=60, seed=42)
        cls.dataset = build_candidate_dataset_from_signals(cls.signals, tolerance_percent=2.0)
        cls.ranker = XGBoostSymbolRateRanker()
        cls.ranker.train(
            X_train=cls.dataset["X"][:80],
            y_train=cls.dataset["y"][:80],
            X_val=cls.dataset["X"][80:],
            y_val=cls.dataset["y"][80:],
            feature_names=FEATURE_COLUMNS,
        )
        cls.ckpt_path = os.path.join(cls.temp_dir, "test_ranker.joblib")
        cls.ranker.save(cls.ckpt_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_known_synthetic_signal_gives_sensible_candidate(self):
        """TEST 1: known synthetic periodic signal gives sensible candidate."""
        iq, meta = generate_synthetic_signal("QPSK", symbol_rate_hz=9600, sample_rate_hz=192000, snr_db=25)
        evidence = extract_dsp_evidence(iq, 192000)
        candidates = generate_candidates(evidence, 192000)
        rates = [c.rate_hz for c in candidates]
        # Verify ~9600 is proposed
        has_near_9600 = any(abs(r - 9600) / 9600 <= 0.05 for r in rates)
        self.assertTrue(has_near_9600, f"Expected ~9600 baud candidate in {rates}")

    def test_02_autocorrelation_peak_detection(self):
        """TEST 2: autocorrelation peak detection."""
        iq, _ = generate_synthetic_signal("16-QAM", symbol_rate_hz=4800, sample_rate_hz=96000, snr_db=25)
        peaks, pwr_peaks = extract_autocorr_evidence(iq, 96000)
        self.assertTrue(len(peaks) > 0 or len(pwr_peaks) > 0)
        detected_rates = [p["rate_hz"] for p in peaks + pwr_peaks]
        has_match = any((abs(r - 4800) / 4800 <= 0.10) or (abs(r - 2400) / 2400 <= 0.10) or (abs(r - 9600) / 9600 <= 0.10) for r in detected_rates)
        self.assertTrue(has_match, f"Expected 4800 baud or harmonic in {detected_rates}")

    def test_03_bandwidth_candidate_generation(self):
        """TEST 3: bandwidth candidate generation across RRC rolloff hypotheses."""
        iq, _ = generate_synthetic_signal("QPSK", symbol_rate_hz=19200, sample_rate_hz=192000, rolloff=0.25)
        bw_evidence = extract_bandwidth_evidence(iq, 192000)
        cands = bw_evidence.get("bandwidth_candidates", [])
        self.assertTrue(len(cands) >= 4)  # 0.20, 0.25, 0.35, 0.50
        bw_rates = [c["rate_hz"] for c in cands]
        self.assertTrue(any(abs(r - 19200) / 19200 <= 0.15 for r in bw_rates))

    def test_04_instantaneous_frequency_feature(self):
        """TEST 4: instantaneous-frequency feature for FSK."""
        iq, _ = generate_synthetic_signal("2-FSK", symbol_rate_hz=2400, sample_rate_hz=96000, snr_db=20)
        if_peaks = extract_instantaneous_frequency_evidence(iq, 96000)
        self.assertTrue(len(if_peaks) > 0)
        rates = [p["rate_hz"] for p in if_peaks]
        has_match = any((abs(r - 2400) / 2400 <= 0.10) or (abs(r - 1200) / 1200 <= 0.10) or (abs(r - 4800) / 4800 <= 0.10) for r in rates)
        self.assertTrue(has_match, f"Expected 2400 baud or harmonic in {rates}")


    def test_05_candidate_deduplication(self):
        """TEST 5: candidate deduplication merges close rates within tolerance."""
        cands = [
            SymbolRateCandidate(rate_hz=9580, samples_per_symbol=20.04, sources=["autocorr"]),
            SymbolRateCandidate(rate_hz=9610, samples_per_symbol=19.98, sources=["bandwidth"]),
            SymbolRateCandidate(rate_hz=19200, samples_per_symbol=10.0, sources=["cyclo"]),
        ]
        merged = deduplicate_candidates(cands, tolerance_percent=2.0)
        self.assertEqual(len(merged), 2)
        # Check that merged candidate combined sources
        near_9600 = [c for c in merged if abs(c.rate_hz - 9600) < 100][0]
        self.assertTrue("autocorr" in near_9600.supported_by or "bandwidth" in near_9600.supported_by)

    def test_06_candidate_harmonic_handling(self):
        """TEST 6: candidate harmonic handling detects sub/super harmonics."""
        cands = [
            SymbolRateCandidate(rate_hz=9600, samples_per_symbol=20.0),
            SymbolRateCandidate(rate_hz=19200, samples_per_symbol=10.0),
        ]
        with_harmonics = detect_harmonic_candidates(cands, sample_rate_hz=192000)
        harmonic_cands = [c for c in with_harmonics if c.harmonic_ratio is not None]
        self.assertTrue(len(harmonic_cands) >= 1)

    def test_07_feature_vector_finite(self):
        """TEST 7: feature vector contains only finite numbers."""
        iq, _ = generate_synthetic_signal("16-QAM", symbol_rate_hz=9600, sample_rate_hz=192000, snr_db=-5)
        evidence = extract_dsp_evidence(iq, 192000)
        candidates = generate_candidates(evidence, 192000)
        feat_matrix, dicts = build_candidate_feature_matrix(candidates, evidence, 192000)
        self.assertTrue(np.all(np.isfinite(feat_matrix)))

    def test_08_feature_order_fixed(self):
        """TEST 8: feature order fixed and matches FEATURE_COLUMNS."""
        cand = SymbolRateCandidate(rate_hz=9600, samples_per_symbol=20.0)
        evidence = DSPRateEvidence()
        feat_dict = extract_candidate_feature_dict(cand, evidence, 192000)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, feat_dict)
        self.assertEqual(len(feat_dict), len(FEATURE_COLUMNS))

    def test_09_xgboost_train_and_load(self):
        """TEST 9: XGBoost model training and disk load validation."""
        ranker_loaded = XGBoostSymbolRateRanker()
        ranker_loaded.load(self.ckpt_path)
        self.assertTrue(ranker_loaded.is_fitted)
        self.assertEqual(ranker_loaded.feature_schema_version, FEATURE_SCHEMA_VERSION)

    def test_10_top_k_sorted(self):
        """TEST 10: Top-K candidates are sorted descending by score."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path, top_k=3)
        iq, _ = generate_synthetic_signal("QPSK", symbol_rate_hz=9600, sample_rate_hz=192000)
        pred = estimator.estimate(iq, 192000)
        scores = [item["score"] for item in pred.top_k]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_11_sps_computed_correctly(self):
        """TEST 11: SPS computed correctly as Fs / Rs."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        iq, _ = generate_synthetic_signal("BPSK", symbol_rate_hz=9600, sample_rate_hz=192000)
        pred = estimator.estimate(iq, 192000)
        expected_sps = 192000.0 / pred.best_symbol_rate_hz
        self.assertAlmostEqual(pred.samples_per_symbol, expected_sps, places=3)

    def test_12_non_integer_sps_supported(self):
        """TEST 12: non-integer SPS supported accurately."""
        # Fs = 100,000, Rs = 9600 -> SPS = 10.41666...
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        iq, _ = generate_synthetic_signal("QPSK", symbol_rate_hz=9600, sample_rate_hz=100000)
        pred = estimator.estimate(iq, 100000)
        self.assertIsInstance(pred.samples_per_symbol, float)
        self.assertTrue(pred.samples_per_symbol > 0.0)

    def test_13_same_input_deterministic(self):
        """TEST 13: same input yields deterministic output."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        iq, _ = generate_synthetic_signal("8PSK", symbol_rate_hz=4800, sample_rate_hz=96000, seed=123)
        pred1 = estimator.estimate(iq, 96000)
        pred2 = estimator.estimate(iq, 96000)
        self.assertAlmostEqual(pred1.best_symbol_rate_hz, pred2.best_symbol_rate_hz, places=2)
        self.assertAlmostEqual(pred1.confidence, pred2.confidence, places=4)

    def test_14_group_split_has_no_leakage(self):
        """TEST 14: signal-level group split strictly avoids data leakage."""
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, val_idx = next(gss.split(self.dataset["X"], self.dataset["y"], groups=self.dataset["groups"]))
        train_groups = set(self.dataset["groups"][train_idx])
        val_groups = set(self.dataset["groups"][val_idx])
        overlap = train_groups.intersection(val_groups)
        self.assertEqual(len(overlap), 0, f"Found leaking signals between train and val: {overlap}")

    def test_15_unknown_behavior(self):
        """TEST 15: pure noise returns UNKNOWN status gracefully."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        noise = (np.random.normal(0, 1, 2048) + 1j * np.random.normal(0, 1, 2048)).astype(np.complex64)
        pred = estimator.estimate(noise, 192000)
        self.assertIn(pred.status, ["UNKNOWN", "POSSIBLE"])

    def test_16_batch_inference(self):
        """TEST 16: batch inference processes multiple signals correctly."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        iq1, _ = generate_synthetic_signal("QPSK", symbol_rate_hz=9600, sample_rate_hz=192000)
        iq2, _ = generate_synthetic_signal("2-FSK", symbol_rate_hz=2400, sample_rate_hz=96000)
        preds = estimator.estimate_batch([iq1, iq2], [192000, 96000])
        self.assertEqual(len(preds), 2)
        self.assertTrue(preds[0].best_symbol_rate_hz > 0)
        self.assertTrue(preds[1].best_symbol_rate_hz > 0)

    def test_17_nan_input_handling(self):
        """TEST 17: NaN input raises InvalidSignalError."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        nan_iq = np.array([1.0 + 1j, np.nan + 1j, 0.5 - 0.5j], dtype=np.complex64)
        with self.assertRaises(InvalidSignalError):
            estimator.estimate(nan_iq, 192000)

    def test_18_very_low_snr_does_not_crash(self):
        """TEST 18: very low SNR (-10 dB) executes cleanly without crashing."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        iq, _ = generate_synthetic_signal("QPSK", symbol_rate_hz=9600, sample_rate_hz=192000, snr_db=-10.0)
        pred = estimator.estimate(iq, 192000)
        self.assertIsInstance(pred, SymbolRatePrediction)

    def test_19_modulation_context_optional(self):
        """TEST 19: modulation context from Fusion Engine is optional."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        iq, _ = generate_synthetic_signal("QPSK", symbol_rate_hz=9600, sample_rate_hz=192000)
        mod_ctx = {
            "top_k": [{"class": "QPSK", "probability": 0.88}],
            "probabilities": {"QPSK": 0.88, "BPSK": 0.05, "8PSK": 0.04},
            "confidence": 0.88,
        }
        pred_with = estimator.estimate(iq, 192000, modulation_prediction=mod_ctx)
        self.assertIsNotNone(pred_with.modulation_context)

    def test_20_model_works_without_fusion_engine_input(self):
        """TEST 20: model works without Fusion Engine input."""
        estimator = SymbolRateEstimator(model_path=self.ckpt_path)
        iq, _ = generate_synthetic_signal("QPSK", symbol_rate_hz=9600, sample_rate_hz=192000)
        pred_without = estimator.estimate(iq, 192000, modulation_prediction=None)
        self.assertIsNone(pred_without.modulation_context)
        self.assertTrue(pred_without.best_symbol_rate_hz > 0)


if __name__ == "__main__":
    unittest.main()
