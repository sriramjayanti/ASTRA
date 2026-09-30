"""
ASTRA Fusion Metrics, Agreement Analysis, and Weight Grid Search Utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

from .models import BranchPrediction, FusionPrediction
from .weighted_fusion import WeightedProbabilityFusion


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Computes Accuracy, Macro/Weighted F1, Precision, Recall, Log Loss, ECE, and Per-Class metrics.
    """
    y_pred = np.argmax(probs, axis=-1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    
    # Calculate log loss safely
    try:
        eps = 1e-12
        p_clipped = np.clip(probs, eps, 1.0)
        p_norm = p_clipped / np.sum(p_clipped, axis=-1, keepdims=True)
        loss = float(log_loss(y_true, p_norm, labels=list(range(len(class_names)))))
    except Exception:
        loss = 0.0

    ece = compute_expected_calibration_error(probs, y_true)

    # Per-class metrics
    per_class_f1 = {}
    for idx, cname in enumerate(class_names):
        mask_true = (y_true == idx)
        mask_pred = (y_pred == idx)
        if np.sum(mask_true) > 0 or np.sum(mask_pred) > 0:
            c_f1 = float(f1_score(mask_true, mask_pred, average="binary", zero_division=0))
            per_class_f1[cname] = c_f1
        else:
            per_class_f1[cname] = 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "log_loss": loss,
        "ece": ece,
        "per_class_f1": per_class_f1,
    }


def compute_agreement_metrics(
    y_true: np.ndarray,
    probs_1d: np.ndarray,
    probs_2d: np.ndarray,
    probs_fused: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes fine-grained agreement / disagreement counts and fusion gains.
    """
    pred_1d = np.argmax(probs_1d, axis=-1)
    pred_2d = np.argmax(probs_2d, axis=-1)
    pred_fused = np.argmax(probs_fused, axis=-1)

    correct_1d = (pred_1d == y_true)
    correct_2d = (pred_2d == y_true)
    correct_fused = (pred_fused == y_true)

    agree = (pred_1d == pred_2d)
    disagree = ~agree

    both_correct = int(np.sum(correct_1d & correct_2d))
    only_1d_correct = int(np.sum(correct_1d & ~correct_2d))
    only_2d_correct = int(np.sum(~correct_1d & correct_2d))
    both_wrong = int(np.sum(~correct_1d & ~correct_2d))

    agree_correct = int(np.sum(agree & correct_1d))
    agree_incorrect = int(np.sum(agree & ~correct_1d))
    disagree_count = int(np.sum(disagree))

    total = len(y_true)
    acc_1d = float(np.mean(correct_1d))
    acc_2d = float(np.mean(correct_2d))
    acc_fused = float(np.mean(correct_fused))

    fusion_gain_acc = acc_fused - acc_1d

    return {
        "total_samples": total,
        "accuracy_1d": acc_1d,
        "accuracy_2d": acc_2d,
        "accuracy_fused": acc_fused,
        "fusion_gain_accuracy": fusion_gain_acc,
        "both_correct": both_correct,
        "only_1d_correct": only_1d_correct,
        "only_2d_correct": only_2d_correct,
        "both_wrong": both_wrong,
        "agree_correct": agree_correct,
        "agree_incorrect": agree_incorrect,
        "disagree_count": disagree_count,
        "agreement_rate": float(np.mean(agree)),
    }


def grid_search_fusion_weights(
    probs_1d: np.ndarray,
    probs_2d: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    weight_steps: int = 11,
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Performs weight grid search w_1d in [0.0, 1.0], w_2d = 1 - w_1d strictly on validation set.
    
    Returns:
        Tuple[best_w1, best_w2, best_metrics, all_grid_results]
    """
    weights = np.linspace(0.0, 1.0, weight_steps)
    all_results: List[Dict[str, Any]] = []

    best_score = -1.0
    best_w1 = 0.70
    best_w2 = 0.30
    best_metrics: Dict[str, Any] = {}

    fusion = WeightedProbabilityFusion(weight_1d=0.70, weight_2d=0.30)

    for w1 in weights:
        w2 = 1.0 - w1
        fusion.set_weights(w1, w2)
        p_fused = fusion.fuse_probabilities(probs_1d, probs_2d)
        metrics = compute_comprehensive_metrics(y_true, p_fused, class_names)
        
        entry = {
            "w_1d": float(w1),
            "w_2d": float(w2),
            **metrics,
        }
        all_results.append(entry)

        # Primary selection on Macro F1, tie-break on Accuracy
        score = metrics["macro_f1"] * 100.0 + metrics["accuracy"]
        if score > best_score:
            best_score = score
            best_w1 = float(w1)
            best_w2 = float(w2)
            best_metrics = entry

    return best_w1, best_w2, best_metrics, all_results
