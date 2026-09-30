"""
ASTRA Random Forest Model Serialization and Checkpoint Management.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
import joblib

from .family_classifier import BroadFamilyClassifier
from .feature_schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, IncompatibleFeatureSchemaError
from .preprocessing import FeaturePreprocessor
from .quality_classifier import SignalQualityClassifier


def save_random_forest_bundle(
    save_path: str,
    family_model: BroadFamilyClassifier,
    quality_model: SignalQualityClassifier,
    preprocessor: FeaturePreprocessor,
    metrics: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves trained family & quality models, imputer, feature schema, and metadata bundle.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    bundle = {
        "family_model": family_model,
        "quality_model": quality_model,
        "preprocessor": preprocessor,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "metrics": metrics or {},
        "metadata": extra_meta or {},
        "model_version": "astra_rf_support_v1.0",
    }
    joblib.dump(bundle, save_path)


def load_random_forest_bundle(
    load_path: str,
) -> Tuple[BroadFamilyClassifier, SignalQualityClassifier, FeaturePreprocessor, Dict[str, Any]]:
    """
    Loads model checkpoint bundle and verifies feature schema version.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")

    bundle = joblib.load(load_path)
    loaded_version = bundle.get("feature_schema_version", "unknown")
    if loaded_version != FEATURE_SCHEMA_VERSION:
        raise IncompatibleFeatureSchemaError(
            f"Feature schema version mismatch: checkpoint has '{loaded_version}', "
            f"runtime expects '{FEATURE_SCHEMA_VERSION}'."
        )

    family_model = bundle["family_model"]
    quality_model = bundle["quality_model"]
    preprocessor = bundle["preprocessor"]
    meta = {
        "metrics": bundle.get("metrics", {}),
        "metadata": bundle.get("metadata", {}),
        "model_version": bundle.get("model_version", "astra_rf_support_v1.0"),
    }
    return family_model, quality_model, preprocessor, meta
