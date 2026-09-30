"""
ASTRA Broad Signal-Family Random Forest Classifier (Task A).
Classifies structured DSP features into FSK, PSK, QAM, and UNKNOWN families.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .feature_schema import FEATURE_COLUMNS, validate_feature_matrix


class BroadFamilyClassifier:
    """
    Supervised Random Forest Classifier for broad signal family identification.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = 14,
        min_samples_split: int = 3,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        class_weight: str = "balanced",
        random_state: int = 42,
        classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.classes = list(classes) if classes else ["FSK", "PSK", "QAM", "UNKNOWN"]
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.feature_columns = list(FEATURE_COLUMNS)

    def fit(self, X: np.ndarray, y: Sequence[str]) -> BroadFamilyClassifier:
        """
        Trains the Broad Family Random Forest.
        """
        X_arr = validate_feature_matrix(X)
        y_arr = np.asarray(y)

        self.model.fit(X_arr, y_arr)
        self.classes = list(self.model.classes_)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns [N, num_classes] probability distribution.
        """
        X_arr = validate_feature_matrix(X)
        if not self.is_fitted:
            # Fallback uniform probabilities
            num_c = len(self.classes)
            return np.ones((len(X_arr), num_c), dtype=np.float32) / num_c

        return self.model.predict_proba(X_arr).astype(np.float32)

    def predict(self, X: np.ndarray) -> List[str]:
        """
        Returns argmax predicted family class per sample.
        """
        X_arr = validate_feature_matrix(X)
        if not self.is_fitted:
            return ["UNKNOWN"] * len(X_arr)
        return list(self.model.predict(X_arr))

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns Gini impurity feature importance for each feature column.
        """
        if not self.is_fitted:
            return {col: 0.0 for col in self.feature_columns}

        imps = self.model.feature_importances_
        return {col: float(imps[i]) for i, col in enumerate(self.feature_columns)}
