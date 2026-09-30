"""
Test suite for Mode A Weighted Probability Fusion.
"""

import numpy as np
import pytest
from astra_fusion.src.models import BranchPrediction, InvalidWeightError
from astra_fusion.src.weighted_fusion import WeightedProbabilityFusion


def _create_mock_branch_pred(
    model_name: str,
    class_names: list,
    probs: list,
    pred_class: str,
    conf: float,
    signal_id: str = "sig_001",
) -> BranchPrediction:
    return BranchPrediction(
        model_name=model_name,
        model_version="1.0.0",
        class_names=class_names,
        logits=[float(np.log(p + 1e-12)) for p in probs],
        probabilities=probs,
        predicted_class=pred_class,
        confidence=conf,
        top_k=[{"rank": 1, "class": pred_class, "probability": conf}],
        feature_embedding=[0.1] * 256,
        source_signal_id=signal_id,
        window_start=0,
        window_end=2048,
    )


def test_5_weighted_probabilities_sum_to_1():
    """TEST 5: Fused output probabilities sum strictly to 1.0."""
    classes = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
    p1 = [0.1, 0.05, 0.7, 0.05, 0.02, 0.03, 0.03, 0.02]
    p2 = [0.05, 0.05, 0.8, 0.02, 0.02, 0.02, 0.02, 0.02]
    
    pred1 = _create_mock_branch_pred("1D", classes, p1, "BPSK", 0.7)
    pred2 = _create_mock_branch_pred("2D", classes, p2, "BPSK", 0.8)

    fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)
    result = fusion.fuse_predictions(pred1, pred2)

    total_prob = sum(result.probabilities.values())
    assert pytest.approx(total_prob, rel=1e-5) == 1.0
    assert result.predicted_class == "BPSK"


def test_6_weights_sum_to_1():
    """TEST 6: Invalid weights raise InvalidWeightError."""
    with pytest.raises(InvalidWeightError):
        WeightedProbabilityFusion(weight_1d=0.8, weight_2d=0.4)  # sum = 1.2

    with pytest.raises(InvalidWeightError):
        WeightedProbabilityFusion(weight_1d=-0.2, weight_2d=1.2)  # negative weight


def test_8_branch_agreement():
    """TEST 8: Both branches agree on predicted class."""
    classes = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
    p1 = [0.02, 0.02, 0.02, 0.85, 0.04, 0.02, 0.02, 0.01]
    p2 = [0.01, 0.01, 0.01, 0.80, 0.12, 0.02, 0.02, 0.01]

    pred1 = _create_mock_branch_pred("1D", classes, p1, "QPSK", 0.85)
    pred2 = _create_mock_branch_pred("2D", classes, p2, "QPSK", 0.80)

    fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)
    result = fusion.fuse_predictions(pred1, pred2)

    assert result.branch_agreement is True
    assert result.predicted_class == "QPSK"
    assert result.status in ["CONFIRMED", "ESTIMATED"]


def test_9_branch_disagreement():
    """TEST 9: Branches disagree on predicted class, resulting in lower certainty."""
    classes = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
    p1 = [0.01, 0.01, 0.01, 0.60, 0.30, 0.03, 0.03, 0.01]  # 1D predicts QPSK
    p2 = [0.01, 0.01, 0.01, 0.30, 0.60, 0.03, 0.03, 0.01]  # 2D predicts 8PSK

    pred1 = _create_mock_branch_pred("1D", classes, p1, "QPSK", 0.60)
    pred2 = _create_mock_branch_pred("2D", classes, p2, "8PSK", 0.60)

    fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)
    result = fusion.fuse_predictions(pred1, pred2)

    assert result.branch_agreement is False
    assert result.status != "CONFIRMED"
    # Fused calculation: QPSK = 0.70*0.60 + 0.30*0.30 = 0.51, 8PSK = 0.70*0.30 + 0.30*0.60 = 0.39
    assert result.predicted_class == "QPSK"
    assert pytest.approx(result.confidence, rel=1e-3) == 0.51


def test_20_deterministic_weighted_fusion():
    """TEST 20: Weighted probability fusion is strictly deterministic."""
    classes = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
    p1 = [0.05] * 8
    p1[3] = 0.65
    p2 = [0.05] * 8
    p2[3] = 0.65

    pred1 = _create_mock_branch_pred("1D", classes, p1, "QPSK", 0.65)
    pred2 = _create_mock_branch_pred("2D", classes, p2, "QPSK", 0.65)

    fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)
    res_a = fusion.fuse_predictions(pred1, pred2)
    res_b = fusion.fuse_predictions(pred1, pred2)

    assert res_a.confidence == res_b.confidence
    assert res_a.predicted_class == res_b.predicted_class
    assert res_a.confidence_margin == res_b.confidence_margin
