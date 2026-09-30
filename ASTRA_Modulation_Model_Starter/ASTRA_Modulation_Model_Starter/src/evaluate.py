import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from dataset import (
    load_config,
    IQSignalDataset,
    IQPreprocessor,
    read_tim_file
)
from model import create_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ASTRA_EVAL")


def evaluate_test_set(
    checkpoint_path: str = "checkpoints/best_model.pt",
    test_manifest_path: str = "checkpoints/manifests/test_manifest.json",
    data_dir_override: Optional[str] = None,
    output_dir: str = "eval_results",
    batch_size: int = 64,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Load Checkpoint
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    classes = ckpt.get("classes", config.get("classes", []))
    
    if not classes:
        raise ValueError("Classes not found in checkpoint or configuration!")

    d_cfg = config.get("dataset", {})
    p_cfg = config.get("preprocessing", {})
    
    tim_dtype = d_cfg.get("tim_dtype", "float32")
    data_dir = data_dir_override or d_cfg.get("data_dir", "data")
    window_size = d_cfg.get("window_size", 2048)

    model = create_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model weights from {ckpt_file}")

    # Centralized Preprocessor
    preprocessor = IQPreprocessor(
        remove_dc=p_cfg.get("remove_dc", True),
        normalization=p_cfg.get("normalization", "rms"),
        epsilon=p_cfg.get("epsilon", 1e-8)
    )

    # 3. Load Test Manifest & Dataset
    test_mf = Path(test_manifest_path)
    if not test_mf.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_mf}")
    
    test_df = pd.read_json(test_mf)
    
    test_dataset = IQSignalDataset(
        manifest_df=test_df,
        data_dir=data_dir,
        tim_dtype=tim_dtype,
        classes=classes,
        window_size=window_size,
        preprocessor=preprocessor,
        use_memmap=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Evaluating {len(test_dataset)} test windows across {len(test_df)} test signal files...")

    all_preds = []
    all_targets = []
    all_snrs = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y, batch_snr in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            all_snrs.extend(batch_snr.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_snrs = np.array(all_snrs)
    all_probs = np.array(all_probs)

    # 4. Global Metrics
    acc = float(accuracy_score(all_targets, all_preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    # 5. Per-Class Metrics
    prec_cls, rec_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(len(classes)), average=None, zero_division=0
    )
    per_class_metrics = {}
    for idx, cname in enumerate(classes):
        per_class_metrics[cname] = {
            "precision": float(prec_cls[idx]),
            "recall": float(rec_cls[idx]),
            "f1_score": float(f1_cls[idx]),
            "support": int(supp_cls[idx])
        }

    # 6. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(classes)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    # Plot Confusion Matrix
    plt.figure(figsize=(9, 7))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("ASTRA Modulation Classification — Normalized Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = ".2f"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm_norm[i, j]:{fmt}}\n({cm[i, j]})"
            plt.text(
                j, i, val_str,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True Modulation", fontsize=11)
    plt.xlabel("Predicted Modulation", fontsize=11)
    plt.tight_layout()
    cm_plot_path = out_path / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # 7. Accuracy vs. SNR Breakdown
    snr_bins = np.unique(np.round(all_snrs, decimals=1))
    snr_bins.sort()
    snr_accuracies = {}
    for snr_val in snr_bins:
        mask = np.isclose(all_snrs, snr_val, atol=0.5)
        if np.sum(mask) > 0:
            bin_acc = accuracy_score(all_targets[mask], all_preds[mask])
            snr_accuracies[float(snr_val)] = {
                "accuracy": float(bin_acc),
                "count": int(np.sum(mask))
            }

    # Plot Accuracy vs SNR
    plt.figure(figsize=(8, 5))
    sorted_snrs = sorted(snr_accuracies.keys())
    acc_values = [snr_accuracies[s]["accuracy"] * 100 for s in sorted_snrs]
    plt.plot(sorted_snrs, acc_values, marker="o", color="#0066cc", linewidth=2, markersize=6)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("ASTRA ResNet-1D: Classification Accuracy vs. SNR (dB)", fontsize=12, fontweight="bold")
    plt.xlabel("In-Band SNR (dB)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.tight_layout()
    snr_plot_path = out_path / "accuracy_vs_snr.png"
    plt.savefig(snr_plot_path, dpi=300)
    plt.close()

    # 8. Consolidate results & export
    results = {
        "summary": {
            "total_test_windows": int(len(test_dataset)),
            "accuracy": float(acc),
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted)
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "accuracy_vs_snr": snr_accuracies
    }

    with open(out_path / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    # 9. Generate Markdown Summary
    md_report = f"""# ASTRA Model 1 Evaluation Report

## Summary Metrics
- **Total Test Windows**: {len(test_dataset):,}
- **Overall Accuracy**: **{acc*100:.2f}%**
- **Macro Precision**: {prec_macro:.4f}
- **Macro Recall**: {rec_macro:.4f}
- **Macro-F1 Score**: **{f1_macro:.4f}**
- **Weighted-F1 Score**: {f1_weighted:.4f}

## Per-Class Metrics
| Modulation | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
"""
    for cname, m in per_class_metrics.items():
        md_report += f"| `{cname}` | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1_score']:.4f} | {m['support']} |\n"

    md_report += f"""
## Accuracy vs. SNR (dB)
| SNR (dB) | Accuracy (%) | Windows |
|----------|--------------|---------|
"""
    for s_val, s_data in sorted(snr_accuracies.items()):
        md_report += f"| {s_val:+.1f} dB | {s_data['accuracy']*100:.2f}% | {s_data['count']} |\n"

    md_report += f"""
## Generated Visualizations
- Confusion Matrix: `{output_dir}/confusion_matrix.png`
- Accuracy vs. SNR Curve: `{output_dir}/accuracy_vs_snr.png`
"""
    with open(out_path / "EVALUATION_SUMMARY.md", "w") as f:
        f.write(md_report)

    logger.info(f"Evaluation complete! Results saved to {out_path}")
    print("\n" + md_report)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA Model 1 Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--test_manifest", type=str, default="checkpoints/manifests/test_manifest.json", help="Path to test manifest JSON")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing .tim files")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory for plots and reports")
    args = parser.parse_args()

    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        test_manifest_path=args.test_manifest,
        data_dir_override=args.data_dir,
        output_dir=args.output_dir
    )
