"""
Evaluation Metrics & Visualization for ASTRA 2D Modulation Classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> Dict:
    """Computes overall and per-class classification metrics."""
    acc = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        targets, predictions, labels=range(len(class_names)), average=None, zero_division=0
    )

    per_class = {}
    for i, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp[i]),
        }

    return {
        "overall_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def analyze_confusion_pairs(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    top_n: int = 5,
) -> List[Dict]:
    """Identifies the most frequent misclassification pairs."""
    cm = confusion_matrix(targets, predictions, labels=range(len(class_names)))
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "error_count": int(cm[i, j]),
                    "error_rate_in_true_class": float(cm[i, j] / (cm[i].sum() + 1e-12)),
                })

    pairs.sort(key=lambda x: x["error_count"], reverse=True)
    return pairs[:top_n]


def plot_confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    output_path: Union[str, Path],
    title: str = "ASTRA 2D Spectrogram Classifier — Confusion Matrix",
) -> None:
    """Plots and saves a normalized confusion matrix heatmap."""
    cm = confusion_matrix(targets, predictions, labels=range(len(class_names)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
            plt.text(
                j, i, val_str,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > cm_norm.max() / 2.0 else "black",
                fontsize=8,
            )

    plt.ylabel("True Modulation", fontsize=11)
    plt.xlabel("Predicted Modulation", fontsize=11)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_accuracy_vs_snr(
    targets: Sequence[int],
    predictions: Sequence[int],
    snrs: Sequence[float],
) -> Dict[float, Dict]:
    """Computes accuracy binned by SNR."""
    all_snrs = np.array(snrs)
    snr_bins = np.unique(np.round(all_snrs))
    snr_bins.sort()
    snr_results = {}
    for sb in snr_bins:
        mask = np.isclose(all_snrs, sb, atol=0.5)
        if np.sum(mask) > 0:
            bin_acc = accuracy_score(np.array(targets)[mask], np.array(predictions)[mask])
            bin_f1 = f1_score(np.array(targets)[mask], np.array(predictions)[mask], average="macro", zero_division=0)
            snr_results[float(sb)] = {
                "accuracy": float(bin_acc),
                "macro_f1": float(bin_f1),
                "count": int(np.sum(mask)),
            }
    return snr_results


def plot_accuracy_vs_snr(
    snr_results: Dict[float, Dict],
    output_path: Union[str, Path],
    title: str = "ASTRA 2D Spectrogram Classifier: Accuracy vs. In-Band SNR",
) -> None:
    """Plots and saves Accuracy vs SNR curve."""
    plt.figure(figsize=(8, 5))
    sorted_snrs = sorted(snr_results.keys())
    acc_vals = [snr_results[s]["accuracy"] * 100 for s in sorted_snrs]
    plt.plot(sorted_snrs, acc_vals, marker="o", color="#800080", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("In-Band SNR (dB)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
