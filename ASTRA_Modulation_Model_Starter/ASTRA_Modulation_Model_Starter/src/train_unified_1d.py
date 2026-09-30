"""
Unified 10-Class 1D ResNet Modulation Classifier Training.

Combines:
1. CSPB Benchmark Dataset (Real PSK / QAM / MSK captures)
2. ASTRA Synthetic Datasets (2-FSK, 4-FSK, framed, FEC-encoded, interleaved, impaired captures)

Target 10-Class Modulation Vocabulary:
["2fsk", "4fsk", "bpsk", "qpsk", "8psk", "dqpsk", "msk", "16qam", "64qam", "256qam"]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    CSPBDatasetCatalog,
    create_cspb_file_splits,
    IQPreprocessor,
    complex_to_channels,
)
from dataset import read_tim_file

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TRAIN_UNIFIED_1D")

UNIFIED_CLASSES = [
    "2fsk",
    "4fsk",
    "bpsk",
    "qpsk",
    "8psk",
    "dqpsk",
    "msk",
    "16qam",
    "64qam",
    "256qam",
]


class UnifiedSignalItem:
    def __init__(
        self,
        source: str,  # "cspb" or "synthetic"
        modulation: str,
        snr_db: float,
        # For synthetic files:
        filepath: Optional[Path] = None,
        # For CSPB zip entries:
        zip_path: Optional[str] = None,
        internal_path: Optional[str] = None,
        raw_scalars: Optional[np.ndarray] = None,
    ):
        self.source = source
        self.modulation = modulation.lower().strip()
        self.snr_db = float(snr_db)
        self.filepath = filepath
        self.zip_path = zip_path
        self.internal_path = internal_path
        self.raw_scalars = raw_scalars


class UnifiedASTRAModDataset(Dataset):
    """
    Unified Dataset for 1D Raw IQ Window Streaming across CSPB and ASTRA Synthetic files.
    """
    def __init__(
        self,
        items: List[UnifiedSignalItem],
        classes: List[str] = UNIFIED_CLASSES,
        window_size: int = 2048,
        stride: int = 2048,
        windows_per_signal: int = 4,
        preprocessor: Optional[IQPreprocessor] = None,
    ):
        self.items = items
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.window_size = window_size
        self.stride = stride
        self.windows_per_signal = windows_per_signal
        self.preprocessor = preprocessor or IQPreprocessor(remove_dc=True, normalization="rms")

        self.windows: List[dict] = []
        self._index_windows()

    def _index_windows(self):
        for item_idx, item in enumerate(self.items):
            if item.modulation not in self.class_to_idx:
                continue
            target = self.class_to_idx[item.modulation]
            for w in range(self.windows_per_signal):
                self.windows.append({
                    "item_idx": item_idx,
                    "target": target,
                    "modulation": item.modulation,
                    "snr_db": item.snr_db,
                    "start_sample": w * self.stride,
                })

        logger.info(f"Indexed {len(self.windows):,} windows (size={self.window_size}) across {len(self.items):,} signals.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        win_info = self.windows[idx]
        item = self.items[win_info["item_idx"]]
        start_s = win_info["start_sample"]
        w_size = self.window_size

        if item.source == "cspb":
            scalars = item.raw_scalars
            start_scalar = start_s * 2
            end_scalar = start_scalar + (w_size * 2)
            if end_scalar <= len(scalars):
                window_scalars = scalars[start_scalar:end_scalar]
            else:
                window_scalars = scalars[: w_size * 2]
            i_samples = window_scalars[0::2]
            q_samples = window_scalars[1::2]
            complex_window = (i_samples + 1j * q_samples).astype(np.complex64)
        else:
            # Synthetic signal file (.iq or .wav)
            if item.raw_scalars is None:
                item.raw_scalars = read_tim_file(item.filepath)
            full_signal = item.raw_scalars
            if len(full_signal) >= start_s + w_size:
                complex_window = full_signal[start_s : start_s + w_size]
            else:
                # Pad or slice from start
                complex_window = np.zeros(w_size, dtype=np.complex64)
                avail = min(len(full_signal), w_size)
                complex_window[:avail] = full_signal[:avail]

        # DC removal and RMS power normalization
        clean_window = self.preprocessor.process(complex_window)

        # Shape [2, N]
        iq_2ch = complex_to_channels(clean_window)
        tensor_x = torch.from_numpy(iq_2ch).float()
        target_y = win_info["target"]
        snr_val = win_info["snr_db"]

        return tensor_x, target_y, snr_val


def load_synthetic_signal_items(dataset_root: Path) -> Tuple[List[UnifiedSignalItem], List[UnifiedSignalItem], List[UnifiedSignalItem]]:
    """Loads and splits synthetic signals by reading manifest split metadata."""
    train_items, val_items, test_items = [], [], []

    manifest_all = dataset_root / "manifests" / "all.csv"
    if not manifest_all.exists():
        logger.warning(f"Manifest not found in: {manifest_all}")
        return train_items, val_items, test_items

    df = pd.read_csv(manifest_all)
    for _, row in df.iterrows():
        mod = str(row["modulation"]).lower().strip()
        snr = float(row.get("inband_snr_db", row.get("snr_db", 10.0)))
        cap_col = "capture_path" if "capture_path" in row else "capture_file_path"
        cap_path_str = str(row[cap_col])
        full_cap_path = Path(cap_path_str)
        if not full_cap_path.is_absolute():
            full_cap_path = Path.cwd() / cap_path_str
        if not full_cap_path.exists():
            full_cap_path = dataset_root / "captures" / Path(cap_path_str).name
        if not full_cap_path.exists():
            continue

        item = UnifiedSignalItem(
            source="synthetic",
            modulation=mod,
            snr_db=snr,
            filepath=full_cap_path,
        )

        split_name = str(row.get("split", "train")).lower()
        if split_name == "train":
            train_items.append(item)
        elif split_name in ("validation", "val"):
            val_items.append(item)
        else:
            test_items.append(item)

    logger.info(f"Loaded synthetic dataset '{dataset_root.name}': {len(train_items)} train, {len(val_items)} val, {len(test_items)} test signals.")
    return train_items, val_items, test_items


def build_unified_dataset_splits(
    cspb_data_dir: Path,
    synthetic_dirs: List[Path],
    max_cspb_signals_per_class: int = 400,
) -> Tuple[List[UnifiedSignalItem], List[UnifiedSignalItem], List[UnifiedSignalItem]]:
    """
    Builds zero-leakage balanced splits combining CSPB and Synthetic datasets.
    """
    # 1. Load CSPB
    catalog = CSPBDatasetCatalog(cspb_data_dir)
    cspb_train_df, cspb_val_df, cspb_test_df = create_cspb_file_splits(
        catalog=catalog,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        max_signals_per_class=max_cspb_signals_per_class,
    )

    import zipfile
    def preload_cspb_df(df: pd.DataFrame) -> List[UnifiedSignalItem]:
        items = []
        grouped = df.groupby("zip_path")
        for zip_p, group in grouped:
            with zipfile.ZipFile(zip_p, "r") as zf:
                for _, row in group.iterrows():
                    raw_b = zf.read(row["internal_path"])
                    arr = np.frombuffer(raw_b, dtype=np.float32)
                    items.append(UnifiedSignalItem(
                        source="cspb",
                        modulation=row["modulation"],
                        snr_db=row["snr_db"],
                        raw_scalars=arr,
                    ))
        return items

    logger.info("Preloading CSPB partitions into memory...")
    train_items = preload_cspb_df(cspb_train_df)
    val_items = preload_cspb_df(cspb_val_df)
    test_items = preload_cspb_df(cspb_test_df)

    # 2. Load Synthetic Datasets (FSK & Framed Signals)
    for syn_dir in synthetic_dirs:
        if syn_dir.exists():
            s_train, s_val, s_test = load_synthetic_signal_items(syn_dir)
            train_items.extend(s_train)
            val_items.extend(s_val)
            test_items.extend(s_test)

    return train_items, val_items, test_items


def train_unified_1d_model(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    cspb_dir = base_dir / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "data"
    starter_dir = base_dir / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter"
    out_dir = starter_dir / "checkpoints"
    eval_dir = starter_dir / "eval_results"

    synthetic_dirs = [
        base_dir / "datasets" / "astra_fsk_enriched_v1",
        base_dir / "datasets" / "astra_training_dataset",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ASTRA: UNIFIED 10-CLASS 1D RESNET TRAINING (CSPB + SYNTHETIC FSK/PSK/QAM)")
    print("=" * 80)

    train_items, val_items, test_items = build_unified_dataset_splits(
        cspb_data_dir=cspb_dir,
        synthetic_dirs=synthetic_dirs,
        max_cspb_signals_per_class=400,
    )

    preprocessor = IQPreprocessor(remove_dc=True, normalization="rms")
    window_size = 2048

    train_ds = UnifiedASTRAModDataset(train_items, classes=UNIFIED_CLASSES, window_size=window_size, windows_per_signal=4, preprocessor=preprocessor)
    val_ds = UnifiedASTRAModDataset(val_items, classes=UNIFIED_CLASSES, window_size=window_size, windows_per_signal=4, preprocessor=preprocessor)
    test_ds = UnifiedASTRAModDataset(test_items, classes=UNIFIED_CLASSES, window_size=window_size, windows_per_signal=4, preprocessor=preprocessor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[UNIFIED DATASET SUMMARY]")
    print(f"  • Total Classes (10): {UNIFIED_CLASSES}")
    print(f"  • Train Windows: {len(train_ds):,} (from {len(train_items):,} signals)")
    print(f"  • Val Windows:   {len(val_ds):,} (from {len(val_items):,} signals)")
    print(f"  • Test Windows:  {len(test_ds):,} (from {len(test_items):,} signals)")
    print(f"  • Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = ResNet1DModClassifier(
        num_classes=len(UNIFIED_CLASSES),
        input_channels=2,
        base_channels=64,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_f1 = -1.0
    best_ckpt_path = out_dir / "best_model_1d_10class_unified.pt"
    t_start = time.time()

    print(f"\n--- Starting Unified 10-Class Training ({epochs} Epochs) ---")
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

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": float(val_acc * 100),
                "val_macro_f1": float(val_f1),
                "classes": UNIFIED_CLASSES,
                "dataset": "CSPB + ASTRA Synthetic (Unified 10-Class)",
                "config": {
                    "classes": UNIFIED_CLASSES,
                    "window_size": window_size,
                    "model_type": "resnet1d",
                }
            }, best_ckpt_path)
            print(f"    -> Saved new best checkpoint (Val Macro-F1: {val_f1:.4f})")

    total_time = time.time() - t_start
    print(f"\n[UNIFIED TRAINING COMPLETED in {total_time/60:.2f} min] Best Val Macro-F1: {best_f1:.4f}")

    # Test Set Evaluation
    print("\n" + "=" * 80)
    print(">>> FINAL EVALUATION ON INDEPENDENT TEST PARTITION (10 CLASSES)")
    print("=" * 80)

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
        test_targets, test_preds, labels=range(len(UNIFIED_CLASSES)), average=None, zero_division=0
    )

    print(f"\n[FINAL 10-CLASS TEST METRICS]")
    print(f"  • Overall Test Accuracy: {test_acc*100:.2f}%")
    print(f"  • Macro-F1 Score:        {test_f1:.4f}")

    print(f"\n  [Per-Class Performance Breakdown]")
    per_class_dict = {}
    for i, cname in enumerate(UNIFIED_CLASSES):
        per_class_dict[cname] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp[i]),
        }
        print(f"    - {cname:8s}: Precision: {prec[i]*100:5.1f}% | Recall: {rec[i]*100:5.1f}% | F1: {f1_cls[i]:.4f} (n={supp[i]})")

    # Confusion Matrix Plot
    cm = confusion_matrix(test_targets, test_preds, labels=range(len(UNIFIED_CLASSES)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("ASTRA 1D ResNet — Unified 10-Class Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(UNIFIED_CLASSES))
    plt.xticks(tick_marks, UNIFIED_CLASSES, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, UNIFIED_CLASSES, fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
            plt.text(
                j, i, val_str,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > cm_norm.max() / 2.0 else "black",
                fontsize=7,
            )
    plt.ylabel("True Modulation", fontsize=11)
    plt.xlabel("Predicted Modulation", fontsize=11)
    plt.tight_layout()
    cm_path = eval_dir / "unified_10class_1d_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Accuracy vs SNR Plot
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
    plt.plot(sorted_snrs, acc_vals, marker="s", color="#008080", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("ASTRA 1D ResNet: Accuracy vs. SNR (10-Class Unified)", fontsize=12, fontweight="bold")
    plt.xlabel("SNR (dB)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.tight_layout()
    snr_path = eval_dir / "unified_10class_1d_accuracy_vs_snr.png"
    plt.savefig(snr_path, dpi=200)
    plt.close()

    # Save JSON Report
    report = {
        "dataset": "CSPB (28 batches) + ASTRA Synthetic (FSK/PSK/QAM/Framed)",
        "model": "ResNet1DModClassifier",
        "num_classes": 10,
        "classes": UNIFIED_CLASSES,
        "total_test_windows": len(test_loader.dataset),
        "overall_accuracy": float(test_acc),
        "macro_f1": float(test_f1),
        "per_class": per_class_dict,
        "accuracy_vs_snr": snr_results,
        "saved_checkpoint": str(best_ckpt_path),
        "confusion_matrix_plot": str(cm_path),
        "accuracy_vs_snr_plot": str(snr_path),
    }
    with open(eval_dir / "unified_10class_1d_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Checkpoint saved: {best_ckpt_path}")
    print(f"[OK] Report saved:     {eval_dir / 'unified_10class_1d_report.json'}")
    print(f"[OK] Plots saved:      {cm_path} and {snr_path}")
    print("=" * 80)


if __name__ == "__main__":
    train_unified_1d_model()
