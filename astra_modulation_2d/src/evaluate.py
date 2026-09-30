"""
ASTRA 2D Spectrogram Classifier Evaluation & Benchmark Engine.

Produces:
1. Standardized classification metrics (Accuracy, Macro-F1, Precision, Recall)
2. Normalized Confusion Matrix heatmap
3. Accuracy vs. In-Band SNR curves
4. Confusion Pair Analysis
5. evaluation_predictions.csv for direct fair comparison with Friend 2D and 1D models
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from astra_modulation_2d.src.utils import load_yaml_config, setup_logger
from astra_modulation_2d.src.model import ASTRASpectrogramCNN
from astra_modulation_2d.src.checkpoint import load_checkpoint
from astra_modulation_2d.src.dataset import SignalWindowRecord, ASTRASpectrogramDataset, DataLoader
from astra_modulation_2d.src.train import build_cspb_records, build_synthetic_records
from astra_modulation_2d.src.metrics import (
    compute_classification_metrics,
    analyze_confusion_pairs,
    plot_confusion_matrix,
    compute_accuracy_vs_snr,
    plot_accuracy_vs_snr,
)

logger = setup_logger("EVALUATE_2D")


def evaluate_model(
    config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """Evaluates trained 2D Spectrogram CNN model on independent test set."""
    cfg = load_yaml_config(config_path)
    ds_cfg = cfg.get("dataset", {})
    split_cfg = cfg.get("split", {})
    prep_cfg = cfg.get("preprocessing", {})
    spec_cfg = cfg.get("spectrogram", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    class_names = [c.lower().strip() for c in ds_cfg.get("class_names", [])]
    window_size = int(ds_cfg.get("window_size", 2048))
    stride = int(ds_cfg.get("stride", 2048))
    windows_per_signal = int(ds_cfg.get("windows_per_signal", 4))

    # Load Test Records
    dtype = ds_cfg.get("type", "cspb")
    if dtype == "cspb":
        data_root = Path(config_path).parent / ds_cfg.get("root", "data")
        split_f = ds_cfg.get("split_file")
        split_p = Path(config_path).parent / split_f if split_f else None
        _, _, test_recs, _ = build_cspb_records(
            data_dir=data_root,
            class_names=class_names,
            split_file=split_p,
            train_ratio=float(split_cfg.get("train", 0.70)),
            val_ratio=float(split_cfg.get("validation", 0.15)),
            test_ratio=float(split_cfg.get("test", 0.15)),
            seed=int(split_cfg.get("seed", 42)),
            windows_per_signal=windows_per_signal,
            window_size=window_size,
            stride=stride,
        )
    else:
        main_root = Path(config_path).parent / ds_cfg.get("root")
        roots = [main_root]
        for add_r in ds_cfg.get("additional_roots", []):
            roots.append(Path(config_path).parent / add_r)
        _, _, test_recs = build_synthetic_records(
            dataset_roots=roots,
            class_names=class_names,
            windows_per_signal=windows_per_signal,
            window_size=window_size,
            stride=stride,
        )

    test_ds = ASTRASpectrogramDataset(test_recs, class_names)
    test_loader = DataLoader(test_ds, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTRASpectrogramCNN(class_names=class_names).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    test_preds, test_targets, test_snrs = [], [], []
    pred_rows = []

    with torch.no_grad():
        for bx, by, b_snr, b_sid, b_start in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            test_preds.extend(preds)
            test_targets.extend(by.numpy())
            test_snrs.extend(b_snr.numpy())

            for i in range(len(preds)):
                row_d = {
                    "source_signal_id": b_sid[i],
                    "window_start": int(b_start[i]),
                    "true_class": class_names[int(by[i])],
                    "predicted_class": class_names[int(preds[i])],
                    "confidence": float(probs[i, preds[i]]),
                    "snr_db": float(b_snr[i]),
                }
                for c_idx, cname in enumerate(class_names):
                    row_d[f"prob_{cname}"] = float(probs[i, c_idx])
                pred_rows.append(row_d)

    # Compute Metrics
    metrics = compute_classification_metrics(test_targets, test_preds, class_names)
    confusion_pairs = analyze_confusion_pairs(test_targets, test_preds, class_names, top_n=5)
    snr_results = compute_accuracy_vs_snr(test_targets, test_preds, test_snrs)

    out_p = Path(output_dir or Path(checkpoint_path).parent)
    out_p.mkdir(parents=True, exist_ok=True)

    # Save Artifacts
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_p / "evaluation_predictions.csv", index=False)

    cm_path = out_p / "confusion_matrix.png"
    plot_confusion_matrix(test_targets, test_preds, class_names, cm_path)

    snr_path = out_p / "accuracy_vs_snr.png"
    plot_accuracy_vs_snr(snr_results, snr_path)

    report = {
        "metrics": metrics,
        "confusion_pairs": confusion_pairs,
        "accuracy_vs_snr": snr_results,
        "test_windows_count": len(test_recs),
        "evaluation_predictions_path": str(out_p / "evaluation_predictions.csv"),
        "confusion_matrix_plot": str(cm_path),
        "accuracy_vs_snr_plot": str(snr_path),
    }

    with open(out_p / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation completed. Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%, Macro-F1: {metrics['macro_f1']:.4f}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA 2D Spectrogram Classifier Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    evaluate_model(args.config, args.checkpoint, args.output_dir)
