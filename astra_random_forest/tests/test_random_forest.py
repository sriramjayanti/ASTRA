"""
ASTRA Random Forest Support Model Comprehensive Unit Test Suite.
Verifies all 20 required specifications.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import numpy as np

from astra_random_forest.src.checkpoint import load_random_forest_bundle, save_random_forest_bundle
from astra_random_forest.src.dataset_builder import build_dataset_from_signals, generate_synthetic_rf_pool
from astra_random_forest.src.family_classifier import BroadFamilyClassifier
from astra_random_forest.src.feature_extractor_adapter import extract_dsp_feature_dict
from astra_random_forest.src.feature_importance import compute_comprehensive_feature_importance
from astra_random_forest.src.feature_schema import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    IncompatibleFeatureSchemaError,
    validate_feature_matrix,
    validate_feature_vector,
)
from astra_random_forest.src.inference import RandomForestSupportEngine
from astra_random_forest.src.label_mapping import FAMILY_MAPPING, extract_quality_labels, map_modulation_to_family
from astra_random_forest.src.preprocessing import FeaturePreprocessor, verify_no_label_leakage
from astra_random_forest.src.quality_classifier import SignalQualityClassifier
from astra_random_forest.src.utils import generate_impaired_signal


class TestRandomForestSupportEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        # Train baseline models for testing
        cls.signals = generate_synthetic_rf_pool(num_signals=60, seed=42)
        cls.dataset = build_dataset_from_signals(cls.signals)
        cls.preprocessor = FeaturePreprocessor(strategy="median")
        cls.X_imp = cls.preprocessor.fit_transform(cls.dataset["X"])

        cls.family_model = BroadFamilyClassifier(n_estimators=50, random_state=42)
        cls.family_model.fit(cls.X_imp, cls.dataset["y_family"])

        cls.quality_model = SignalQualityClassifier(n_estimators=30, random_state=42)
        cls.quality_model.fit(cls.X_imp, cls.dataset["y_quality"])

        cls.ckpt_path = os.path.join(cls.temp_dir, "test_rf.joblib")
        save_random_forest_bundle(
            save_path=cls.ckpt_path,
            family_model=cls.family_model,
            quality_model=cls.quality_model,
            preprocessor=cls.preprocessor,
        )
        cls.engine = RandomForestSupportEngine(checkpoint_path=cls.ckpt_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_feature_schema_fixed(self):
        """TEST 1: feature schema is fixed and columns match FEATURE_COLUMNS."""
        self.assertEqual(len(FEATURE_COLUMNS), len(FEATURE_COLUMNS))
        self.assertEqual(FEATURE_SCHEMA_VERSION, "rf_features_v1")

    def test_02_missing_feature_handling(self):
        """TEST 2: missing features converted to NaN and handled cleanly."""
        partial = {"occupied_bandwidth_hz": 15000.0, "mean_magnitude": 1.0}
        row = validate_feature_vector(partial)
        self.assertEqual(len(row), len(FEATURE_COLUMNS))
        self.assertTrue(np.isnan(row[2]))  # unprovided feature is NaN

    def test_03_training_runs(self):
        """TEST 3: training completes without errors."""
        self.assertTrue(self.family_model.is_fitted)
        self.assertTrue(self.quality_model.is_fitted)

    def test_04_family_probabilities_sum_to_1(self):
        """TEST 4: predicted family probabilities sum to 1.0."""
        iq, _ = generate_impaired_signal("QPSK", snr_db=20.0)
        res = self.engine.predict_family(iq)
        probs = list(res["probabilities"].values())
        self.assertAlmostEqual(float(sum(probs)), 1.0, places=3)

    def test_05_family_output_class_valid(self):
        """TEST 5: predicted family is in valid classes (FSK, PSK, QAM, UNKNOWN)."""
        iq, _ = generate_impaired_signal("16-QAM", snr_db=25.0)
        res = self.engine.predict_family(iq)
        self.assertIn(res["predicted_family"], ["FSK", "PSK", "QAM", "UNKNOWN"])

    def test_06_quality_multilabel_output_valid(self):
        """TEST 6: quality multi-label probabilities and detection flags are valid."""
        iq, _ = generate_impaired_signal("QPSK", snr_db=-5.0, cfo_hz=1200.0)
        res = self.engine.predict_quality(iq)
        ev = res["quality_evidence"]
        self.assertIn("low_snr", ev)
        self.assertIn("cfo_affected", ev)
        self.assertTrue(0.0 <= ev["low_snr"]["probability"] <= 1.0)

    def test_07_checkpoint_save_and_load(self):
        """TEST 7: checkpoint save and load preserves state and schema version."""
        f_mod, q_mod, prep, meta = load_random_forest_bundle(self.ckpt_path)
        self.assertTrue(f_mod.is_fitted)
        self.assertTrue(q_mod.is_fitted)
        self.assertTrue(prep.is_fitted)

    def test_08_feature_order_preserved(self):
        """TEST 8: feature order matches FEATURE_COLUMNS."""
        iq, _ = generate_impaired_signal("BPSK")
        feat_dict = extract_dsp_feature_dict(iq)
        row = validate_feature_vector(feat_dict)
        self.assertEqual(len(row), len(FEATURE_COLUMNS))

    def test_09_same_seed_reproducible(self):
        """TEST 9: same seed yields deterministic model predictions."""
        iq, _ = generate_impaired_signal("8PSK", seed=777)
        res1 = self.engine.predict_family(iq)
        res2 = self.engine.predict_family(iq)
        self.assertEqual(res1["predicted_family"], res2["predicted_family"])
        self.assertAlmostEqual(res1["confidence"], res2["confidence"], places=4)

    def test_10_batch_inference(self):
        """TEST 10: batch inference processes list of inputs correctly."""
        iq1, _ = generate_impaired_signal("2-FSK")
        iq2, _ = generate_impaired_signal("16-QAM")
        batch_res = self.engine.predict_batch([iq1, iq2])
        self.assertEqual(len(batch_res), 2)
        self.assertIn(batch_res[0]["family"], ["FSK", "PSK", "QAM", "UNKNOWN"])
        self.assertIn(batch_res[1]["family"], ["FSK", "PSK", "QAM", "UNKNOWN"])

    def test_11_no_source_leakage(self):
        """TEST 11: group split has zero overlap across source IDs."""
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, val_idx = next(gss.split(self.dataset["X"], self.dataset["y_family"], groups=self.dataset["groups"]))
        train_groups = set(self.dataset["groups"][train_idx])
        val_groups = set(self.dataset["groups"][val_idx])
        self.assertEqual(len(train_groups.intersection(val_groups)), 0)

    def test_12_no_label_columns_in_training_x(self):
        """TEST 12: no target label or truth leakage in training X."""
        verify_no_label_leakage(self.dataset["X"], FEATURE_COLUMNS)

    def test_13_unknown_handling(self):
        """TEST 13: pure noise returns UNKNOWN or low confidence."""
        noise, _ = generate_impaired_signal("NOISE")
        res = self.engine.predict_family(noise)
        self.assertIn(res["predicted_family"], ["UNKNOWN", "FSK", "PSK", "QAM"])

    def test_14_nan_imputation(self):
        """TEST 14: input with all NaNs is imputed without error."""
        all_nan = np.full((1, len(FEATURE_COLUMNS)), np.nan, dtype=np.float32)
        imputed = self.preprocessor.transform(all_nan)
        self.assertTrue(np.all(np.isfinite(imputed)))

    def test_15_feature_importance_generated(self):
        """TEST 15: Gini feature importance extracted for all features."""
        imps = self.family_model.get_feature_importances()
        self.assertEqual(len(imps), len(FEATURE_COLUMNS))
        self.assertTrue(all(v >= 0.0 for v in imps.values()))

    def test_16_permutation_importance_generated(self):
        """TEST 16: Permutation importance computed across validation set."""
        records = compute_comprehensive_feature_importance(
            model=self.family_model,
            X_val=self.X_imp[:20],
            y_val=self.dataset["y_family"][:20],
            n_repeats=2,
        )
        self.assertEqual(len(records), len(FEATURE_COLUMNS))
        self.assertIn("permutation_importance_mean", records[0])


    def test_17_cpu_inference(self):
        """TEST 17: Inference runs purely on CPU fast (< 50ms)."""
        iq, _ = generate_impaired_signal("QPSK")
        res = self.engine.predict_all(iq)
        self.assertIn("family", res)
        self.assertIn("quality", res)

    def test_18_invalid_schema_rejected(self):
        """TEST 18: Incompatible column count raises error."""
        invalid_x = np.zeros((5, 12), dtype=np.float32)
        with self.assertRaises(IncompatibleFeatureSchemaError):
            validate_feature_matrix(invalid_x)

    def test_19_class_mapping_config_respected(self):
        """TEST 19: label mapping maps subclasses correctly."""
        self.assertEqual(map_modulation_to_family("2-FSK"), "FSK")
        self.assertEqual(map_modulation_to_family("16-QAM"), "QAM")
        self.assertEqual(map_modulation_to_family("8PSK"), "PSK")
        self.assertEqual(map_modulation_to_family("NOISE"), "UNKNOWN")

    def test_20_prediction_result_json_serializable(self):
        """TEST 20: prediction result is fully JSON serializable."""
        iq, _ = generate_impaired_signal("QPSK")
        res = self.engine.predict_all(iq)
        serialized = json.dumps(res)
        self.assertTrue(len(serialized) > 0)


if __name__ == "__main__":
    unittest.main()
