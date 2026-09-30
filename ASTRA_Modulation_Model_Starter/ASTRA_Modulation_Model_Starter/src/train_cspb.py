"""
Train Ready 1D ResNet Model on Complete CSPB.ML.2018R2 Dataset (28 Batches).

Outputs:
1. Trained model checkpoint: checkpoints/best_model_cspb_baseline.pt
2. Evaluation metrics: eval_results/cspb_baseline_report.json
3. Confusion matrix plot: eval_results/cspb_confusion_matrix.png
4. Accuracy vs SNR plot: eval_results/cspb_accuracy_vs_snr.png
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from model import ResNet1DModClassifier
from cspb_loader import (
    CSPB_CLASSES,
    CSPBDatasetCatalog,
    create_cspb_file_splits,
    get_cspb_dataloader,
    IQPreprocessor,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TRAIN_CSPB")


def train_cspb_baseline(
    data_dir: str = "data",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_signals_per_class: int = 500,  # 500 signals/class * 8 classes * 4 windows = 16,000 windows for fast benchmark
    output_dir: str = "checkpoints",
    eval_dir: str = "eval_results",
):
    src_dir = Path(__file__).parent
    starter_dir = src_dir.parent
    data_path = starter_dir / data_dir
    out_path = starter_dir / output_dir
    eval_path = starter_dir / eval_dir

    out_path.mkdir(parents=True, exist_ok=True)
    eval_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ASTRA: TRAINING 1D RESNET MODEL ON CSPB.ML.2018R2 BENCHMARK")
    print("=" * 80)

    # 1. Catalog & Index all 28 Batches
    catalog = CSPBDatasetCatalog(data_path)

    # 2. File-Level Split (Zero Leakage)
    train_df, val_df, test_df = create_cspb_file_splits(
        catalog=catalog,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        max_signals_per_class=max_signals_per_class,
    )

    # 3. DataLoaders
    preprocessor = IQPreprocessor(remove_dc=True, normalization="rms")
    window_size = 2048

    train_loader = get_cspb_dataloader(
        train_df,
        classes=CSPB_CLASSES,
        batch_size=batch_size,
        shuffle=True,
        window_size=window_size,
        windows_per_signal=4,
        preprocessor=preprocessor,
    )

    val_loader = get_cspb_dataloader(
        val_df,
        classes=CSPB_CLASSES,
        batch_size=batch_size,
        shuffle=False,
        window_size=window_size,
        windows_per_signal=4,
        preprocessor=preprocessor,
    )

    test_loader = get_cspb_dataloader(
        test_df,
        classes=CSPB_CLASSES,
        batch_size=batch_size,
        shuffle=False,
        window_size=window_size,
        windows_per_signal=4,
        preprocessor=preprocessor,
    )

    print(f"\n[DATASET PARTITION SUMMARY]")
    print(f"  • Train Windows: {len(train_loader.dataset):,} (from {len(train_df):,} complete signal files)")
    print(f"  • Val Windows:   {len(val_loader.dataset):,} (from {len(val_df):,} complete signal files)")
    print(f"  • Test Windows:  {len(test_loader.dataset):,} (from {len(test_df):,} complete signal files)")

    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  • Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = ResNet1DModClassifier(
        num_classes=len(CSPB_CLASSES),
        input_channels=2,
        base_channels=64,
        dropout=0.2,
    ).to(device)

    print(f"  • 1D ResNet Initialized: {model.count_parameters():,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Training Loop
    best_f1 = -1.0
    best_ckpt_path = out_path / "best_model_cspb_baseline.pt"
    history = []
    t_start = time.time()

    print(f"\n--- Starting 1D ResNet Training ({epochs} Epochs) ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(by.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        dur = time.time() - t0
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({dur:4.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.1f}% F1: {val_f1:.4f} | LR: {cur_lr:.1e}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": cur_lr,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": float(val_acc * 100),
                "val_macro_f1": float(val_f1),
                "classes": CSPB_CLASSES,
                "dataset": "CSPB.ML.2018R2",
                "config": {
                    "classes": CSPB_CLASSES,
                    "window_size": window_size,
                    "model_type": "resnet1d",
                }
            }, best_ckpt_path)
            print(f"    -> Saved new best checkpoint (Val Macro-F1: {val_f1:.4f})")

    total_time = time.time() - t_start
    print(f"\n[TRAINING COMPLETED in {total_time/60:.2f} min] Best Val Macro-F1: {best_f1:.4f}")

    # 6. Test Set Evaluation
    print("\n" + "-" * 80)
    print(">>> EVALUATING ON INDEPENDENT TEST SET (ZERO LEAKAGE)")
    print("-" * 80)

    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_preds, test_targets, test_snrs = [], [], []
    with torch.no_grad():
        for bx, by, b_snr in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_targets.extend(by.numpy())
            test_snrs.extend(b_snr.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        test_targets, test_preds, labels=range(len(CSPB_CLASSES)), average=None, zero_division=0
    )

    print(f"\n[FINAL CSPB TEST METRICS]")
    print(f"  • Overall Test Accuracy: {test_acc*100:.2f}%")
    print(f"  • Macro-F1 Score:        {test_f1:.4f}")

    print(f"\n  [Per-Class Breakdown]")
    per_class_dict = {}
    for i, cname in enumerate(CSPB_CLASSES):
        per_class_dict[cname] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp[i]),
        }
        print(f"    - {cname:8s}: Precision: {prec[i]*100:5.1f}% | Recall: {rec[i]*100:5.1f}% | F1: {f1_cls[i]:.4f} (n={supp[i]})")

    # 7. Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds, labels=range(len(CSPB_CLASSES)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("ASTRA 1D ResNet — CSPB.ML.2018R2 Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(CSPB_CLASSES))
    plt.xticks(tick_marks, CSPB_CLASSES, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, CSPB_CLASSES, fontsize=10)

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
    cm_path = eval_path / "cspb_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 8. Accuracy vs SNR
    all_snrs = np.array(test_snrs)
    snr_bins = np.unique(np.round(all_snrs))
    snr_bins.sort()
    snr_results = {}
    for sb in snr_bins:
        mask = np.isclose(all_snrs, sb, atol=0.5)
        if np.sum(mask) > 0:
            bin_acc = accuracy_score(np.array(test_targets)[mask], np.array(test_preds)[mask])
            snr_results[float(sb)] = {"accuracy": float(bin_acc), "count": int(np.sum(mask))}

    plt.figure(figsize=(8, 5))
    sorted_snrs = sorted(snr_results.keys())
    acc_vals = [snr_results[s]["accuracy"] * 100 for s in sorted_snrs]
    plt.plot(sorted_snrs, acc_vals, marker="o", color="#0066cc", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("ASTRA 1D ResNet: Accuracy vs. In-Band SNR (CSPB)", fontsize=12, fontweight="bold")
    plt.xlabel("In-Band SNR (dB)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.tight_layout()
    snr_path = eval_path / "cspb_accuracy_vs_snr.png"
    plt.savefig(snr_path, dpi=200)
    plt.close()

    # Save JSON Report
    report = {
        "dataset": "CSPB.ML.2018R2 (28 batches)",
        "model": "ResNet1DModClassifier",
        "total_test_windows": len(test_loader.dataset),
        "overall_accuracy": float(test_acc),
        "macro_f1": float(test_f1),
        "per_class": per_class_dict,
        "accuracy_vs_snr": snr_results,
        "saved_checkpoint": str(best_ckpt_path),
        "confusion_matrix_plot": str(cm_path),
        "accuracy_vs_snr_plot": str(snr_path),
    }
    with open(eval_path / "cspb_baseline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Checkpoint saved: {best_ckpt_path}")
    print(f"[OK] Report saved:     {eval_path / 'cspb_baseline_report.json'}")
    print(f"[OK] Plots saved:      {cm_path} and {snr_path}")
    print("\n" + "=" * 80)
    print("CSPB BENCHMARK TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    train_cspb_baseline()
