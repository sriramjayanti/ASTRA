"""
ASTRA Signal-Quality and Corruption Multi-Label Classifier (Task B).
Detects LOW_SNR, CLIPPED, MULTIPATH, and CFO_AFFECTED channel impairments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from .feature_schema import FEATURE_COLUMNS, validate_feature_matrix


class SignalQualityClassifier:
    """
    Multi-label Random Forest Classifier predicting independent channel impairment probabilities.
    """

    def __init__(
        self,
        labels: Optional[Sequence[str]] = None,
        n_estimators: int = 200,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        self.labels = list(labels) if labels else ["low_snr", "clipped", "multipath", "cfo_affected"]
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model = MultiOutputClassifier(base_rf)
        self.is_fitted = False
        self.feature_columns = list(FEATURE_COLUMNS)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SignalQualityClassifier:
        """
        Trains the multi-label impairment classifier.
        y shape: [N_samples, N_labels] binary {0, 1} matrix.
        """
        X_arr = validate_feature_matrix(X)
        y_arr = np.asarray(y, dtype=np.int32)

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns a dict mapping label_name -> array of probabilities [N_samples].
        """
        X_arr = validate_feature_matrix(X)
        if not self.is_fitted:
            # Fallback 0.5 probability for each label
            return {lbl: np.full(len(X_arr), 0.5, dtype=np.float32) for lbl in self.labels}

        # MultiOutputClassifier.predict_proba returns list of length N_labels
        # each element has shape [N_samples, 2] (prob of class 0 and 1)
        probs_list = self.model.predict_proba(X_arr)
        result: Dict[str, np.ndarray] = {}

        for idx, lbl in enumerate(self.labels):
            if idx < len(probs_list):
                p = probs_list[idx]
                if p.ndim == 2 and p.shape[1] == 2:
                    result[lbl] = p[:, 1].astype(np.float32)
                elif p.ndim == 2 and p.shape[1] == 1:
                    result[lbl] = np.zeros(len(X_arr), dtype=np.float32)
                else:
                    result[lbl] = p.flatten().astype(np.float32)
            else:
                result[lbl] = np.zeros(len(X_arr), dtype=np.float32)

        return result

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Returns binary [N_samples, N_labels] prediction matrix.
        """
        probs_dict = self.predict_proba(X)
        cols = [probs_dict[lbl] >= threshold for lbl in self.labels]
        return np.column_stack(cols).astype(np.int32)
