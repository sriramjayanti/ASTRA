"""
ASTRA Random Forest Preprocessing and Imputation Pipeline.
Handles missing feature values reliably and prevents label leakage.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from sklearn.impute import SimpleImputer

from .feature_schema import FEATURE_COLUMNS, IncompatibleFeatureSchemaError, validate_feature_matrix


class FeaturePreprocessor:
    """
    Fits and applies median imputation across the 36-feature schema.
    """

    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy, missing_values=np.nan)
        self.is_fitted = False
        self.feature_columns = list(FEATURE_COLUMNS)

    def fit(self, X: np.ndarray) -> FeaturePreprocessor:
        """
        Fits the imputer on training feature matrix.
        """
        X_arr = validate_feature_matrix(X)
        self.imputer.fit(X_arr)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Imputes NaNs in the feature matrix.
        """
        X_arr = validate_feature_matrix(X)
        if not self.is_fitted:
            # If not fitted yet, replace remaining NaNs with 0.0 fallback
            return np.nan_to_num(X_arr, nan=0.0)
        return self.imputer.transform(X_arr).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits imputer and transforms X.
        """
        self.fit(X)
        return self.transform(X)


def verify_no_label_leakage(feature_matrix: np.ndarray, feature_names: Sequence[str]) -> None:
    """
    Asserts that no label, truth, or metadata columns exist in the feature set.
    """
    forbidden_terms = [
        "label", "target", "modulation", "truth", "ground_truth",
        "family", "class", "filename", "signal_id", "source_id"
    ]
    for name in feature_names:
        lower = name.lower()
        for term in forbidden_terms:
            if lower == term or (lower.startswith(term) and lower != "clipping_ratio"):
                raise IncompatibleFeatureSchemaError(
                    f"FATAL: Label leakage detected in feature '{name}'!"
                )
