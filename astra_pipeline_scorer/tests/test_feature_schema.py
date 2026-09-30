"""
test_feature_schema.py
Unit tests for feature schema consistency, ordering, and extraction.
"""

import pytest
import numpy as np
from astra_pipeline_scorer.src.feature_schema import (
    PIPELINE_FEATURE_COLUMNS,
    PIPELINE_FEATURE_SCHEMA_VERSION,
    get_feature_schema,
)
from astra_pipeline_scorer.src.feature_builder import PipelineFeatureBuilder


def test_feature_schema_fixed():
    """Verify feature schema version and columns are fixed and unique."""
    schema = get_feature_schema()
    assert schema["schema_version"] == "pipeline_features_v1"
    assert len(PIPELINE_FEATURE_COLUMNS) >= 60
    assert len(PIPELINE_FEATURE_COLUMNS) == len(set(PIPELINE_FEATURE_COLUMNS))


def test_feature_extraction_from_candidate_path():
    """Verify feature builder extracts 1D vector matching exact schema length."""
    builder = PipelineFeatureBuilder()
    mock_candidate = {
        "modulation": "QPSK",
        "symbol_rate_hz": 9600.0,
        "interleaver_family": "block",
        "fec_family": "convolutional",
        "stage5_metrics": {"modulation_probability": 0.85, "symbol_rate_score": 0.90},
        "stage6_sync_metrics": {"timing_lock_score": 0.92, "evm_percent": 8.5},
        "stage7_demod_metrics": {"evm_percent": 8.5, "demod_quality_score": 0.88},
        "stage8_interleaver_metrics": {"interleaver_structural_score": 0.80},
        "stage9_fec_metrics": {"decoder_success": True, "fec_quality_score": 0.95},
        "stage10_validation_metrics": {"crc_pass": 1.0, "overall_validation_score": 0.92},
    }

    vec = builder.extract_feature_vector(mock_candidate)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == len(PIPELINE_FEATURE_COLUMNS)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_missing_feature_handling():
    """Verify empty dictionary receives valid default fallback values with no NaNs."""
    builder = PipelineFeatureBuilder()
    vec = builder.extract_feature_vector({})
    assert len(vec) == len(PIPELINE_FEATURE_COLUMNS)
    assert not np.isnan(vec).any()
