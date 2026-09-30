"""
evaluation.py
Signal-level ranking and candidate classification evaluation metrics for ASTRA Stage 11.
Computes Top-1/Top-3/Top-5 accuracy, MRR, Candidate Recall@K, ROC-AUC, PR-AUC, Brier score, and ECE.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from .models import EvaluationReport


def compute_expected_calibration_error(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=np.int32)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_prob)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / n
            ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_signal_level_ranking(
    signal_ids: List[str],
    y_scores: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    Group candidates by signal_id, sort descending by score, and evaluate ranking metrics:
    - Top-1 Accuracy
    - Top-3 Accuracy
    - Top-5 Accuracy
    - MRR (Mean Reciprocal Rank)
    - Candidate Recall (Was positive candidate present?)
    - Conditional Top-1 Accuracy (Accuracy among signals where truth was present)
    """
    signal_ids = list(signal_ids)
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.int32).ravel()

    # Group by signal
    signal_map: Dict[str, List[Tuple[float, int]]] = {}
    for sid, score, label in zip(signal_ids, y_scores, y_true):
        if sid not in signal_map:
            signal_map[sid] = []
        signal_map[sid].append((float(score), int(label)))

    total_signals = len(signal_map)
    signals_with_truth = 0
    top1_correct_count = 0
    top3_correct_count = 0
    top5_correct_count = 0
    reciprocal_ranks = []

    for sid, cand_list in signal_map.items():
        # Sort candidates descending by predicted score
        cand_list.sort(key=lambda x: x[0], reverse=True)
        has_positive = any(label == 1 for _, label in cand_list)

        if has_positive:
            signals_with_truth += 1
            # Find 1-indexed rank of first correct candidate
            rank_of_truth = None
            for idx, (_, label) in enumerate(cand_list):
                if label == 1:
                    rank_of_truth = idx + 1
                    break

            if rank_of_truth is not None:
                reciprocal_ranks.append(1.0 / float(rank_of_truth))
                if rank_of_truth == 1:
                    top1_correct_count += 1
                if rank_of_truth <= 3:
                    top3_correct_count += 1
                if rank_of_truth <= 5:
                    top5_correct_count += 1
            else:
                reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(0.0)

    top1_acc = top1_correct_count / max(1, total_signals)
    top3_acc = top3_correct_count / max(1, total_signals)
    top5_acc = top5_correct_count / max(1, total_signals)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    conditional_top1 = top1_correct_count / max(1, signals_with_truth)
    recall = signals_with_truth / max(1, total_signals)

    return {
        "total_signals": total_signals,
        "signals_with_truth": signals_with_truth,
        "candidate_recall": recall,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "mrr": mrr,
        "conditional_top1_accuracy": conditional_top1,
    }


def evaluate_pipeline_model(
    signal_ids: List[str],
    y_scores: np.ndarray,
    y_true: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> EvaluationReport:
    """
    Produce comprehensive EvaluationReport object.
    """
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64).ravel() if y_probs is not None else y_scores

    ranking_metrics = evaluate_signal_level_ranking(signal_ids, y_scores, y_true)

    # Candidate-level classification metrics
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    roc_auc = 0.0
    pr_auc = 0.0
    loss_val = 0.0
    brier = 0.0
    ece_val = 0.0

    if n_pos > 0 and n_neg > 0:
        roc_auc = float(roc_auc_score(y_true, y_probs))
        pr_auc = float(average_precision_score(y_true, y_probs))
        loss_val = float(log_loss(y_true, np.clip(y_probs, 1e-6, 1.0 - 1e-6)))
        brier = float(brier_score_loss(y_true, y_probs))
        ece_val = compute_expected_calibration_error(y_probs, y_true)

    return EvaluationReport(
        total_signals=ranking_metrics["total_signals"],
        total_candidates=len(y_scores),
        positive_candidates=n_pos,
        negative_candidates=n_neg,
        candidate_recall_top1=ranking_metrics["top1_accuracy"],
        candidate_recall_top3=ranking_metrics["top3_accuracy"],
        candidate_recall_top5=ranking_metrics["top5_accuracy"],
        mrr=ranking_metrics["mrr"],
        conditional_top1_acc=ranking_metrics["conditional_top1_accuracy"],
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        log_loss=loss_val,
        brier_score=brier,
        ece=ece_val,
    )
