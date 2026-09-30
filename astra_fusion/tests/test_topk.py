"""
Test suite for Top-K candidate list generation and sorting.
"""

import numpy as np
import pytest
from astra_fusion.src.models import BranchPrediction
from astra_fusion.src.weighted_fusion import WeightedProbabilityFusion


def test_7_top_k_sorted():
    """TEST 7: Top-K candidates are correctly ranked and sorted in descending probability order."""
    classes = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
    p1 = [0.05, 0.05, 0.05, 0.60, 0.20, 0.03, 0.01, 0.01]
    p2 = [0.05, 0.05, 0.05, 0.50, 0.30, 0.03, 0.01, 0.01]

    pred1 = BranchPrediction(
        model_name="1D", model_version="1.0", class_names=classes, logits=p1,
        probabilities=p1, predicted_class="QPSK", confidence=0.60, top_k=[],
    )
    pred2 = BranchPrediction(
        model_name="2D", model_version="1.0", class_names=classes, logits=p2,
        probabilities=p2, predicted_class="QPSK", confidence=0.50, top_k=[],
    )

    fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30, top_k=3)
    result = fusion.fuse_predictions(pred1, pred2)

    assert len(result.top_k) == 3
    # Check monotonic descending probability
    probs = [item["probability"] for item in result.top_k]
    assert probs == sorted(probs, reverse=True)
    assert result.top_k[0]["class"] == "QPSK"
    assert result.top_k[1]["class"] == "8PSK"
