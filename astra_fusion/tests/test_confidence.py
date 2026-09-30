"""
Test suite for Confidence margin, Entropy calculation, and ASTRA Statusing.
"""

import numpy as np
import pytest
from astra_fusion.src.confidence import ConfidenceCalculator


def test_10_confidence_margin():
    """TEST 10: Margin equals top1_probability - top2_probability."""
    probs = np.array([0.70, 0.20, 0.05, 0.05], dtype=np.float32)
    top1, top2, margin = ConfidenceCalculator.compute_margin(probs)
    assert pytest.approx(top1, rel=1e-4) == 0.70
    assert pytest.approx(top2, rel=1e-4) == 0.20
    assert pytest.approx(margin, rel=1e-4) == 0.50


def test_11_entropy_finite():
    """TEST 11: Normalized Shannon entropy is finite, between 0.0 and 1.0."""
    # Deterministic vector
    p_det = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ent_det = ConfidenceCalculator.compute_entropy(p_det)
    assert np.isfinite(ent_det)
    assert pytest.approx(ent_det, abs=1e-5) == 0.0

    # Uniform vector
    p_uni = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    ent_uni = ConfidenceCalculator.compute_entropy(p_uni)
    assert np.isfinite(ent_uni)
    assert pytest.approx(ent_uni, abs=1e-4) == 1.0


def test_12_unknown_status():
    """TEST 12: Low confidence or high uncertainty produces UNKNOWN status."""
    calc = ConfidenceCalculator(
        confirmed_threshold=0.85,
        estimated_threshold=0.50,
        possible_threshold=0.30,
        confirmed_margin=0.20,
    )

    # Low confidence scenario
    status, reason = calc.determine_status(
        predicted_class="QPSK",
        confidence=0.20,
        confidence_margin=0.02,
        branch_agreement=False,
        entropy=0.95,
        is_unknown_class=False,
    )
    assert status == "UNKNOWN"
    assert reason in ["low_confidence", "branch_conflict", "high_uncertainty"]


def test_13_explicit_unknown_class():
    """TEST 13: Predicted class 'Unknown' outputs UNKNOWN status with explicit reason."""
    calc = ConfidenceCalculator()
    status, reason = calc.determine_status(
        predicted_class="Unknown",
        confidence=0.98,
        confidence_margin=0.90,
        branch_agreement=True,
        entropy=0.05,
        is_unknown_class=True,
    )
    assert status == "UNKNOWN"
    assert reason == "explicit_unknown_class"
