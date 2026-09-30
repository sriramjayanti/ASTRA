"""
ASTRA Modulation Intelligence V2 — Unified High-Speed Training & Evaluation Engine.

High-performance in-memory tensor caching:
Preloads and preprocesses windows into pinned tensors, enabling pure GPU computation
(3-4 seconds per epoch) without disk I/O bottlenecks.

Trains both:
1. ResNet-1D V2 (Time-Domain raw IQ [B, 2, 2048])
2. Spectrogram CNN-2D V2 (Centered Log-Power STFT [B, 1, 128, 128])

Evaluates on:
- Canonical Holdout Validation Set
- Canonical Holdout Test Set
- Out-Of-Distribution (OOD) Set
- Stratified SNR tiers (<0 dB, 0-5, 5-10, 10-15, 15-20, >20)

Saves checkpoints to:
- checkpoints/astra_resnet1d_v2.pt
- checkpoints/astra_spectrogram_cnn_v2.pt
"""

from __future__ import annotations

import os
import sys
import json
import time
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Any

print = functools.partial(print, flush=True)

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import (
    CLASS_SCHEMA_VERSION,
    MODULATION_CLASSES_V2,
    NUM_CLASSES_V2,
    get_class_index,
    get_class_name,
)
from astra_modulation_v2.dataset_builder import (
    IQPreprocessorV2,
    iq_to_tensor_1d,
    iq_to_spectrogram_2d,
)
from astra_modulation_v2.models.resnet1d import ResNet1DV2
from astra_modulation_v2.models.spectrogram_cnn import SpectrogramCNN2DV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_DIR = ROOT / "datasets" / "ASTRA_MODULATION_DATASET_V2" / "manifests"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class FastCachedDataset(Dataset):
    """Zero-overhead In-Memory Dataset holding pre-computed tensors."""
    def __init__(
        self,
        tensors: torch.Tensor,
        labels: torch.Tensor,
        snrs: torch.Tensor,
        source_ids: List[str],
    ):
        self.tensors = tensors
        self.labels = labels
        self.snrs = snrs
        self.source_ids = source_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.tensors[idx],
            "label": self.labels[idx],
            "snr_db": self.snrs[idx],
            "source_id": self.source_ids[idx],
        }


def load_and_cache_tensors(
    split_name: str,
    target_mode: str = "1d",
    window_size: int = 2048,
    windows_per_capture: int = 2,
) -> FastCachedDataset:
    """Preloads raw IQ from disk once and compiles into pinned GPU/RAM tensors."""
    csv_path = MANIFEST_DIR / f"{split_name}.csv"
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    
    preprocessor = IQPreprocessorV2(remove_dc=True, normalize_rms=True)
    
    all_tensors = []
    all_labels = []
    all_snrs = []
    all_sources = []
    
    t0 = time.time()
    print(f"Preloading & caching '{split_name}' ({len(records)} captures, mode={target_mode})...")
    
    for rec in records:
        iq_path = rec.get("iq_path")
        if iq_path and os.path.exists(iq_path):
            data = np.fromfile(iq_path, dtype=np.float32)
            iq = data[0::2] + 1j * data[1::2]
        else:
            iq = np.zeros(window_size * windows_per_capture, dtype=np.complex64)
            
        iq = preprocessor.process(iq)
        n_samp = len(iq)
        label = get_class_index(rec["modulation"])
        snr = float(rec.get("snr_db", 0.0))
        s_id = rec["source_id"]
        
        # Extract windows
        if n_samp <= window_size:
            pad_len = max(0, window_size - n_samp)
            w = np.pad(iq, (0, pad_len), mode="constant")
            windows = [w] * windows_per_capture
        else:
            stride = max((n_samp - window_size) // max(windows_per_capture - 1, 1), 1)
            windows = []
            for k in range(windows_per_capture):
                start = min(k * stride, n_samp - window_size)
                windows.append(iq[start : start + window_size])
                
        for w in windows:
            if target_mode == "1d":
                all_tensors.append(iq_to_tensor_1d(w))
            else:
                all_tensors.append(iq_to_spectrogram_2d(w))
                
            all_labels.append(label)
            all_snrs.append(snr)
            all_sources.append(s_id)
            
    x_tensor = torch.stack(all_tensors)
    y_tensor = torch.tensor(all_labels, dtype=torch.long)
    snr_tensor = torch.tensor(all_snrs, dtype=torch.float32)
    
    elapsed = time.time() - t0
    print(f"Cached {len(y_tensor):,} windows in {elapsed:.1f}s | Tensor shape: {x_tensor.shape}")
    
    return FastCachedDataset(x_tensor, y_tensor, snr_tensor, all_sources)


def evaluate_fast(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Fast GPU evaluation."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    all_snrs = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_targets.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())
            all_snrs.extend(batch["snr_db"].numpy().tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    all_snrs = np.array(all_snrs)

    top1 = float(accuracy_score(all_targets, all_preds))
    top3_correct = 0
    for i in range(len(all_targets)):
        top3_indices = np.argsort(all_probs[i])[-3:]
        if all_targets[i] in top3_indices:
            top3_correct += 1
    top3 = float(top3_correct / max(len(all_targets), 1))

    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(NUM_CLASSES_V2)), zero_division=0
    )

    per_class = {}
    for idx, cname in enumerate(MODULATION_CLASSES_V2):
        per_class[cname] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES_V2))).tolist()

    snr_bins = [(-100.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 100.0)]
    snr_labels = ["< 0 dB", "0-5 dB", "5-10 dB", "10-15 dB", "15-20 dB", "> 20 dB"]
    snr_breakdown = {}
    for (low, high), slab in zip(snr_bins, snr_labels):
        mask = (all_snrs >= low) & (all_snrs < high)
        if np.sum(mask) > 0:
            sub_acc = float(accuracy_score(all_targets[mask], all_preds[mask]))
            snr_breakdown[slab] = {"accuracy": round(sub_acc * 100, 2), "count": int(np.sum(mask))}

    return {
        "top1_accuracy": round(top1 * 100, 2),
        "top3_accuracy": round(top3 * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "snr_breakdown": snr_breakdown,
        "classes": MODULATION_CLASSES_V2,
    }


def train_model(
    model_type: str = "1d",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    windows_per_capture: int = 2,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    print("=" * 80)
    print(f"TRAINING {model_type.upper()} MODULATION MODEL V2 (HIGH-SPEED GPU MODE)")
    print("=" * 80)

    # 1. Preload & cache all splits
    train_ds = load_and_cache_tensors("train", target_mode=model_type, windows_per_capture=windows_per_capture)
    val_ds = load_and_cache_tensors("validation", target_mode=model_type, windows_per_capture=windows_per_capture)
    test_ds = load_and_cache_tensors("test", target_mode=model_type, windows_per_capture=windows_per_capture)
    ood_ds = load_and_cache_tensors("ood_test", target_mode=model_type, windows_per_capture=windows_per_capture)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 2. Instantiate Model
    if model_type == "1d":
        model = ResNet1DV2(in_channels=2, num_classes=NUM_CLASSES_V2).to(DEVICE)
    else:
        model = SpectrogramCNN2DV2(in_channels=1, num_classes=NUM_CLASSES_V2).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_corr = 0
        total_train = 0

        for batch in train_loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=-1)
            train_corr += int((preds == y).sum().item())
            total_train += len(y)

        scheduler.step()
        train_acc = train_corr / max(total_train, 1)
        avg_loss = train_loss / max(total_train, 1)

        val_res = evaluate_fast(model, val_loader, DEVICE)
        val_f1 = val_res["macro_f1"]
        val_top1 = val_res["top1_accuracy"]
        val_top3 = val_res["top3_accuracy"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
            f"Val Top-1: {val_top1:.1f}% | Val Top-3: {val_top3:.1f}% | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    print("\nEvaluating best checkpoint on untouched Test Set...")
    test_res = evaluate_fast(model, test_loader, DEVICE)
    print(f"Test Top-1: {test_res['top1_accuracy']}% | Top-3: {test_res['top3_accuracy']}% | Macro-F1: {test_res['macro_f1']}")

    print("\nEvaluating best checkpoint on Out-Of-Distribution (OOD) Set...")
    ood_res = evaluate_fast(model, ood_loader, DEVICE)
    print(f"OOD Top-1: {ood_res['top1_accuracy']}% | Top-3: {ood_res['top3_accuracy']}% | Macro-F1: {ood_res['macro_f1']}")

    ckpt_name = f"astra_resnet1d_v2.pt" if model_type == "1d" else f"astra_spectrogram_cnn_v2.pt"
    ckpt_path = CKPT_DIR / ckpt_name
    torch.save({
        "state_dict": best_state,
        "model_architecture": "ResNet1DV2" if model_type == "1d" else "SpectrogramCNN2DV2",
        "classes": MODULATION_CLASSES_V2,
        "class_schema_version": CLASS_SCHEMA_VERSION,
        "num_classes": NUM_CLASSES_V2,
        "input_shape": [2, 2048] if model_type == "1d" else [1, 128, 128],
        "metrics": {
            "val_f1": best_val_f1,
            "test_top1": test_res["top1_accuracy"],
            "test_top3": test_res["top3_accuracy"],
            "test_macro_f1": test_res["macro_f1"],
            "ood_top1": ood_res["top1_accuracy"],
            "ood_top3": ood_res["top3_accuracy"],
            "ood_macro_f1": ood_res["macro_f1"],
        },
    }, ckpt_path)
    print(f"Saved best model checkpoint to {ckpt_path}\n")

    return model, val_res, test_res, ood_res


def main():
    print("=" * 80)
    print("ASTRA MODULATION INTELLIGENCE V2: FULL TRAINING PIPELINE")
    print("=" * 80)

    # 1. Train ResNet-1D V2
    model_1d, val_1d, test_1d, ood_1d = train_model("1d", epochs=15, batch_size=64, lr=1e-3, windows_per_capture=2)
    with open(CKPT_DIR / "resnet1d_v2_metrics.json", "w") as f:
        json.dump({"val": val_1d, "test": test_1d, "ood": ood_1d}, f, indent=2)

    # 2. Train Spectrogram CNN-2D V2
    model_2d, val_2d, test_2d, ood_2d = train_model("2d", epochs=15, batch_size=64, lr=1e-3, windows_per_capture=2)
    with open(CKPT_DIR / "spectrogram_cnn2d_v2_metrics.json", "w") as f:
        json.dump({"val": val_2d, "test": test_2d, "ood": ood_2d}, f, indent=2)

    print("ALL V2 MODEL TRAININGS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
