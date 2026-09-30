"""
test_calibration.py
Unit tests for probability calibration.
"""

import pytest
import numpy as np
from astra_pipeline_scorer.src.calibration import ScoreCalibrator
from astra_pipeline_scorer.src.evaluation import compute_expected_calibration_error


def test_score_calibrator_isotonic():
    raw_scores = np.array([0.1, 0.2, 0.35, 0.7, 0.85, 0.95])
    y_true = np.array([0, 0, 0, 1, 1, 1])

    calib = ScoreCalibrator(method="isotonic")
    calib.fit(raw_scores, y_true)
    assert calib.is_fitted is True

    calibrated = calib.calibrate(raw_scores)
    assert len(calibrated) == len(raw_scores)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_expected_calibration_error():
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])
    ece = compute_expected_calibration_error(y_prob, y_true)
    assert 0.0 <= ece <= 1.0
