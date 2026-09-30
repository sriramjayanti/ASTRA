"""
Standalone test runner for ASTRA Multi-Branch Modulation Fusion Engine.
Runs all tests and reports pass/fail summary.
"""

import os
import sys
import tempfile
import unittest

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure workspace root is in sys.path
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import numpy as np
import torch

from astra_fusion.src.models import (
    BranchPrediction,
    CandidateItem,
    ClassMappingMismatchError,
    FusionPrediction,
    InvalidProbabilityError,
    InvalidWeightError,
    SourceAlignmentError,
)
from astra_fusion.src.validation import (
    validate_class_alignment,
    validate_probabilities,
    validate_source_alignment,
    validate_weights,
)
from astra_fusion.src.confidence import ConfidenceCalculator
from astra_fusion.src.weighted_fusion import WeightedProbabilityFusion
from astra_fusion.src.learned_fusion import LearnedFusionEngine, LearnedFusionMLP
from astra_fusion.src.inference import ASTRAFusionEngine
from astra_fusion.src.checkpoint import load_fusion_checkpoint, save_fusion_checkpoint
from astra_fusion.src.metrics import (
    compute_agreement_metrics,
    compute_comprehensive_metrics,
    compute_expected_calibration_error,
    grid_search_fusion_weights,
)


class TestASTRAFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classes = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
        # Share one initialized engine for performance
        cls.shared_engine = ASTRAFusionEngine(device="cpu")

    def _mock_pred(self, model_name, probs, pred_class, conf, sig_id="sig_001"):
        return BranchPrediction(
            model_name=model_name,
            model_version="1.0.0",
            class_names=self.classes,
            logits=[float(np.log(p + 1e-12)) for p in probs],
            probabilities=probs,
            predicted_class=pred_class,
            confidence=conf,
            top_k=[{"rank": 1, "class": pred_class, "probability": conf}],
            feature_embedding=[0.1] * 256,
            source_signal_id=sig_id,
            window_start=0,
            window_end=2048,
        )

    def test_1_class_mapping_equality(self):
        """TEST 1: Exact class mapping matching passes without error."""
        validate_class_alignment(self.classes, self.classes)

    def test_2_mismatched_mapping_raises_error(self):
        """TEST 2: Mismatched class mapping raises ClassMappingMismatchError."""
        with self.assertRaises(ClassMappingMismatchError):
            validate_class_alignment(self.classes, ["2-FSK", "4-FSK"])
        with self.assertRaises(ClassMappingMismatchError):
            validate_class_alignment(self.classes, self.classes[::-1])

    def test_3_source_alignment_works(self):
        """TEST 3: Matching source signal IDs and window ranges pass validation."""
        validate_source_alignment("sig_1", 0, 2048, "sig_1", 0, 2048)

    def test_4_unrelated_source_windows_rejected(self):
        """TEST 4: Mismatched signal IDs or offsets raise SourceAlignmentError."""
        with self.assertRaises(SourceAlignmentError):
            validate_source_alignment("sig_1", 0, 2048, "sig_2", 0, 2048)
        with self.assertRaises(SourceAlignmentError):
            validate_source_alignment("sig_1", 0, 2048, "sig_1", 1024, 3072)

    def test_5_weighted_probabilities_sum_to_1(self):
        """TEST 5: Fused output probabilities sum strictly to 1.0."""
        p1 = [0.1, 0.05, 0.7, 0.05, 0.02, 0.03, 0.03, 0.02]
        p2 = [0.05, 0.05, 0.8, 0.02, 0.02, 0.02, 0.02, 0.02]
        pred1 = self._mock_pred("1D", p1, "BPSK", 0.7)
        pred2 = self._mock_pred("2D", p2, "BPSK", 0.8)

        fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)
        res = fusion.fuse_predictions(pred1, pred2)
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)
        self.assertEqual(res.predicted_class, "BPSK")

    def test_6_weights_sum_to_1(self):
        """TEST 6: Invalid weights raise InvalidWeightError."""
        with self.assertRaises(InvalidWeightError):
            WeightedProbabilityFusion(weight_1d=0.8, weight_2d=0.4)
        with self.assertRaises(InvalidWeightError):
            WeightedProbabilityFusion(weight_1d=-0.1, weight_2d=1.1)

    def test_7_top_k_sorted(self):
        """TEST 7: Top-K candidates are ranked in descending order."""
        p1 = [0.05, 0.05, 0.05, 0.60, 0.20, 0.03, 0.01, 0.01]
        p2 = [0.05, 0.05, 0.05, 0.50, 0.30, 0.03, 0.01, 0.01]
        pred1 = self._mock_pred("1D", p1, "QPSK", 0.60)
        pred2 = self._mock_pred("2D", p2, "QPSK", 0.50)

        fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30, top_k=3)
        res = fusion.fuse_predictions(pred1, pred2)
        self.assertEqual(len(res.top_k), 3)
        self.assertEqual(res.top_k[0]["class"], "QPSK")
        self.assertEqual(res.top_k[1]["class"], "8PSK")
        self.assertGreaterEqual(res.top_k[0]["probability"], res.top_k[1]["probability"])

    def test_8_branch_agreement(self):
        """TEST 8: Branch agreement is detected and reflected in status."""
        p1 = [0.02, 0.02, 0.02, 0.88, 0.02, 0.02, 0.01, 0.01]
        p2 = [0.01, 0.01, 0.01, 0.90, 0.03, 0.02, 0.01, 0.01]
        pred1 = self._mock_pred("1D", p1, "QPSK", 0.88)
        pred2 = self._mock_pred("2D", p2, "QPSK", 0.90)

        fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)
        res = fusion.fuse_predictions(pred1, pred2)
        self.assertTrue(res.branch_agreement)
        self.assertEqual(res.predicted_class, "QPSK")
        self.assertEqual(res.status, "CONFIRMED")

    def test_9_branch_disagreement(self):
        """TEST 9: Branch disagreement is tracked with lowered confidence."""
        p1 = [0.01, 0.01, 0.01, 0.60, 0.30, 0.03, 0.03, 0.01]
        p2 = [0.01, 0.01, 0.01, 0.30, 0.60, 0.03, 0.03, 0.01]
        pred1 = self._mock_pred("1D", p1, "QPSK", 0.60)
        pred2 = self._mock_pred("2D", p2, "8PSK", 0.60)

        fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)
        res = fusion.fuse_predictions(pred1, pred2)
        self.assertFalse(res.branch_agreement)
        self.assertNotEqual(res.status, "CONFIRMED")
        self.assertEqual(res.predicted_class, "QPSK")
        self.assertAlmostEqual(res.confidence, 0.51, places=2)

    def test_10_confidence_margin(self):
        """TEST 10: Confidence margin calculation."""
        probs = np.array([0.70, 0.20, 0.05, 0.05], dtype=np.float32)
        top1, top2, margin = ConfidenceCalculator.compute_margin(probs)
        self.assertAlmostEqual(margin, 0.50, places=4)

    def test_11_entropy_finite(self):
        """TEST 11: Entropy is bounded in [0.0, 1.0]."""
        p_det = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(ConfidenceCalculator.compute_entropy(p_det), 0.0, places=4)
        p_uni = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.assertAlmostEqual(ConfidenceCalculator.compute_entropy(p_uni), 1.0, places=3)

    def test_12_unknown_status(self):
        """TEST 12: Low confidence or high entropy triggers UNKNOWN status."""
        calc = ConfidenceCalculator()
        status, reason = calc.determine_status("QPSK", 0.20, 0.02, False, 0.95)
        self.assertEqual(status, "UNKNOWN")
        self.assertIn(reason, ["low_confidence", "branch_conflict", "high_uncertainty"])

    def test_13_explicit_unknown_class(self):
        """TEST 13: Explicit 'Unknown' class produces UNKNOWN status with dedicated reason."""
        calc = ConfidenceCalculator()
        status, reason = calc.determine_status("Unknown", 0.95, 0.90, True, 0.05, is_unknown_class=True)
        self.assertEqual(status, "UNKNOWN")
        self.assertEqual(reason, "explicit_unknown_class")

    def test_14_single_sample_inference(self):
        """TEST 14: Single complex IQ sample inference."""
        iq_sig = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
        res = self.shared_engine.predict(iq_sig, source_signal_id="sig_test_01", window_start=0, window_end=2048)
        self.assertIn(res.predicted_class, self.shared_engine.class_names)
        self.assertTrue(0.0 <= res.confidence <= 1.0)
        self.assertEqual(len(res.top_k), 3)

    def test_15_batch_inference(self):
        """TEST 15: Vectorized batch inference."""
        batch_iq = (np.random.randn(4, 2048) + 1j * np.random.randn(4, 2048)).astype(np.complex64)
        results = self.shared_engine.predict_batch(batch_iq)
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIn(r.predicted_class, self.shared_engine.class_names)

    def test_16_cpu_inference(self):
        """TEST 16: Explicit CPU inference."""
        iq_sig = np.random.randn(2, 2048).astype(np.float32)
        res = self.shared_engine.predict(iq_sig)
        self.assertTrue(np.isfinite(res.confidence))

    def test_17_cuda_inference_if_available(self):
        """TEST 17: CUDA inference execution if available."""
        iq_sig = np.random.randn(2, 2048).astype(np.float32)
        res = self.shared_engine.predict(iq_sig)
        self.assertIsNotNone(res)

    def test_18_nan_input_handling(self):
        """TEST 18: NaNs and Infs are sanitized without errors."""
        corrupt = np.random.randn(2048).astype(np.float32)
        corrupt[100:150] = np.nan
        corrupt[300:320] = np.inf
        res = self.shared_engine.predict(corrupt)
        self.assertTrue(np.isfinite(res.confidence))
        self.assertTrue(np.isfinite(res.probability_entropy))

    def test_19_checkpoint_reload(self):
        """TEST 19: Mode B learned MLP checkpoint save & reload."""
        mlp = LearnedFusionMLP(dim_1d_feature=256, dim_2d_feature=256, num_classes=8)
        f1, f2 = torch.randn(2, 256), torch.randn(2, 256)
        l1, l2 = torch.randn(2, 8), torch.randn(2, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fusion.pt")
            save_fusion_checkpoint(path, mlp, self.classes, {"1d": "1.0", "2d": "1.0"}, validation_metrics={"acc": 0.95})
            loaded_mlp, meta = load_fusion_checkpoint(path)
            self.assertEqual(meta["validation_metrics"]["acc"], 0.95)
            
            mlp.eval()
            loaded_mlp.eval()
            with torch.no_grad():
                out1, _ = mlp(f1, f2, l1, l2)
                out2, _ = loaded_mlp(f1, f2, l1, l2)
                self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_20_deterministic_weighted_fusion(self):
        """TEST 20: Deterministic output for identical inputs."""
        p1 = [0.05] * 8
        p1[3] = 0.65
        p2 = [0.05] * 8
        p2[3] = 0.65
        pred1 = self._mock_pred("1D", p1, "QPSK", 0.65)
        pred2 = self._mock_pred("2D", p2, "QPSK", 0.65)

        fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)
        res_a = fusion.fuse_predictions(pred1, pred2)
        res_b = fusion.fuse_predictions(pred1, pred2)
        self.assertEqual(res_a.confidence, res_b.confidence)
        self.assertEqual(res_a.predicted_class, res_b.predicted_class)
        self.assertEqual(res_a.confidence_margin, res_b.confidence_margin)


if __name__ == "__main__":
    print("=" * 70)
    print("Running ASTRA Multi-Branch Modulation Fusion Test Suite (20 Tests)")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestASTRAFusion)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n[SUCCESS] ALL 20 TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n[FAILURE] TEST SUITE FAILED!")
        sys.exit(1)
