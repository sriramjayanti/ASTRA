"""
calibration.py
Probability calibration module for ASTRA Stage 11 — Pipeline Scoring Model.
Supports Isotonic Regression and Platt scaling (Sigmoid) on validation splits.
"""

from typing import Optional, Union, List, Any
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    """
    Fits a calibrator on raw model scores or probabilities from the validation split.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._calibrator = None
        self.is_fitted = False

    def fit(self, y_raw_scores: np.ndarray, y_true: np.ndarray) -> "ScoreCalibrator":
        """Fit calibrator on validation split scores."""
        y_raw = np.asarray(y_raw_scores, dtype=np.float64).ravel()
        y_t = np.asarray(y_true, dtype=np.int32).ravel()

        if len(y_raw) == 0 or len(np.unique(y_t)) < 2:
            self.is_fitted = False
            return self

        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._calibrator.fit(y_raw, y_t)
        else: # sigmoid / platt
            self._calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self._calibrator.fit(y_raw.reshape(-1, 1), y_t)

        self.is_fitted = True
        return self

    def calibrate(self, raw_scores: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Calibrate raw model scores into true probabilities [0.0, 1.0]."""
        arr = np.asarray(raw_scores, dtype=np.float64).ravel()
        if not self.is_fitted or self._calibrator is None:
            return np.clip(arr, 0.0, 1.0)

        if self.method == "isotonic":
            calibrated = self._calibrator.predict(arr)
        else:
            calibrated = self._calibrator.predict_proba(arr.reshape(-1, 1))[:, 1]

        return np.clip(calibrated, 0.0, 1.0)
