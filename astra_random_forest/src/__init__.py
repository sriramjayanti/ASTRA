"""
ASTRA Random Forest Support Engine Package.
"""

from .checkpoint import load_random_forest_bundle, save_random_forest_bundle
from .family_classifier import BroadFamilyClassifier
from .feature_extractor_adapter import extract_dsp_feature_dict
from .feature_importance import compute_comprehensive_feature_importance, export_feature_importance_csv
from .feature_schema import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    IncompatibleFeatureSchemaError,
    validate_feature_matrix,
    validate_feature_vector,
)
from .inference import RandomForestSupportEngine
from .label_mapping import FAMILY_MAPPING, extract_quality_labels, map_modulation_to_family
from .preprocessing import FeaturePreprocessor, verify_no_label_leakage
from .quality_classifier import SignalQualityClassifier

__all__ = [
    "RandomForestSupportEngine",
    "BroadFamilyClassifier",
    "SignalQualityClassifier",
    "FeaturePreprocessor",
    "FEATURE_COLUMNS",
    "FEATURE_SCHEMA_VERSION",
    "FAMILY_MAPPING",
    "IncompatibleFeatureSchemaError",
    "extract_dsp_feature_dict",
    "validate_feature_vector",
    "validate_feature_matrix",
    "map_modulation_to_family",
    "extract_quality_labels",
    "verify_no_label_leakage",
    "compute_comprehensive_feature_importance",
    "export_feature_importance_csv",
    "save_random_forest_bundle",
    "load_random_forest_bundle",
]
