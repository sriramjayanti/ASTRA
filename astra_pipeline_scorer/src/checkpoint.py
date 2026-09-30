"""
checkpoint.py
Model checkpoint packaging and persistence for ASTRA Stage 11.
Saves and loads the trained XGBoost model, probability calibrator, feature schema, and metadata.
"""

from typing import Any, Dict, Optional, Tuple
import os
import json
import joblib
import xgboost as xgb

from .feature_schema import PIPELINE_FEATURE_COLUMNS, PIPELINE_FEATURE_SCHEMA_VERSION
from .calibration import ScoreCalibrator


def save_pipeline_model_package(
    model: xgb.XGBClassifier,
    calibrator: Optional[ScoreCalibrator],
    save_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save complete model package directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Save XGBoost model
    model_path = os.path.join(save_dir, "xgboost_model.json")
    model.save_model(model_path)

    # 2. Save Calibrator
    if calibrator is not None and calibrator.is_fitted:
        calib_path = os.path.join(save_dir, "calibrator.joblib")
        joblib.dump(calibrator, calib_path)

    # 3. Save Feature Schema
    schema_info = {
        "schema_version": PIPELINE_FEATURE_SCHEMA_VERSION,
        "feature_count": len(PIPELINE_FEATURE_COLUMNS),
        "columns": list(PIPELINE_FEATURE_COLUMNS),
    }
    with open(os.path.join(save_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2)

    # 4. Save Metadata
    meta = metadata or {}
    meta["model_version"] = "astra_pipeline_xgb_v1"
    meta["feature_schema_version"] = PIPELINE_FEATURE_SCHEMA_VERSION
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return save_dir


def load_pipeline_model_package(
    package_dir: str,
) -> Tuple[xgb.XGBClassifier, Optional[ScoreCalibrator], Dict[str, Any]]:
    """
    Load trained model package from directory.
    Verifies schema version compatibility before loading.
    """
    if not os.path.isdir(package_dir):
        raise FileNotFoundError(f"Model package directory not found: {package_dir}")

    # Verify schema version
    schema_path = os.path.join(package_dir, "feature_schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_info = json.load(f)
            ver = schema_info.get("schema_version")
            if ver != PIPELINE_FEATURE_SCHEMA_VERSION:
                raise ValueError(
                    f"Incompatible feature schema version: {ver} (expected {PIPELINE_FEATURE_SCHEMA_VERSION})"
                )

    # Load XGBoost model
    model_path = os.path.join(package_dir, "xgboost_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Load Calibrator
    calibrator = None
    calib_path = os.path.join(package_dir, "calibrator.joblib")
    if os.path.exists(calib_path):
        calibrator = joblib.load(calib_path)

    # Load Metadata
    meta_path = os.path.join(package_dir, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, calibrator, metadata
