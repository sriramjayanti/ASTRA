"""
ASTRA Random Forest Feature Schema and Order Definition.
Fixed schema ensuring 100% feature order parity between training and inference.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

FEATURE_SCHEMA_VERSION = "rf_features_v1"

# 36 Deterministic Engineered DSP Features
FEATURE_COLUMNS: List[str] = [
    # 1. Spectral Features
    "occupied_bandwidth_hz",
    "spectral_centroid_hz",
    "spectral_spread_hz",
    "spectral_flatness",
    "spectral_rolloff_hz",
    "psd_variance",
    "peak_to_average_power_ratio",

    # 2. Amplitude & Envelope Features
    "mean_magnitude",
    "variance_magnitude",
    "rms_amplitude",
    "skewness_magnitude",
    "kurtosis_magnitude",
    "crest_factor",
    "envelope_variance",

    # 3. Phase Features
    "mean_phase_diff",
    "variance_phase_diff",
    "circular_variance",
    "phase_concentration",

    # 4. Instantaneous Frequency Features
    "inst_freq_mean_hz",
    "inst_freq_std_hz",
    "inst_freq_kurtosis",
    "dominant_frequency_states",

    # 5. IQ Statistical Moments & Cumulants
    "i_variance",
    "q_variance",
    "iq_covariance",
    "circularity_coefficient",
    "real_imag_correlation",
    "cumulant_c40_mag",
    "cumulant_c42_mag",

    # 6. Temporal & Autocorrelation Features
    "autocorr_max_peak_lag",
    "autocorr_max_peak_score",
    "zero_crossing_rate",

    # 7. Constellation & Energy Level Features
    "constellation_cluster_count",
    "radial_ring_count",
    "constant_modulus_variance",

    # 8. Signal Quality Features
    "estimated_snr_db",
    "clipping_ratio",
    "cfo_magnitude_hz",
]


class IncompatibleFeatureSchemaError(Exception):
    """Raised when an input feature vector does not conform to the schema."""
    pass


def validate_feature_vector(feature_dict: Dict[str, Any]) -> List[float]:
    """
    Validates and extracts an ordered list of float features according to FEATURE_COLUMNS.
    Fills missing features with NaN so the pipeline imputer can handle them.
    """
    row = []
    for col in FEATURE_COLUMNS:
        val = feature_dict.get(col, np.nan)
        if val is None or not isinstance(val, (int, float, np.number)):
            val = np.nan
        elif not math.isfinite(val):
            val = np.nan
        row.append(float(val))
    return row


def validate_feature_matrix(
    feature_matrix: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Validates shape and column count of feature matrix.
    """
    arr = np.asarray(feature_matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != len(FEATURE_COLUMNS):
        raise IncompatibleFeatureSchemaError(
            f"Expected {len(FEATURE_COLUMNS)} feature columns, but received {arr.shape[1]}."
        )

    if feature_names is not None:
        if list(feature_names) != FEATURE_COLUMNS:
            raise IncompatibleFeatureSchemaError(
                "Feature names or order mismatch against FEATURE_SCHEMA_VERSION = 'rf_features_v1'."
            )

    return arr
