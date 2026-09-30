"""
train_modulation_models_v3.py
High-performance training engine for ASTRA Stage 3 Modulation Intelligence V3.
Trains and hardens:
1. ResNet-1D Time-Domain Classifier ([B, 2, 2048])
2. Spectrogram CNN-2D Classifier ([B, 1, 128, 128])
on channel-augmented data (2-3 tap multipath, CFO +-4 kHz, IQ imbalance, noise).
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import (
    CLASS_SCHEMA_VERSION,
    MODULATION_CLASSES_V2,
    NUM_CLASSES_V2,
    CLASS_TO_IDX_V2,
    IDX_TO_CLASS_V2,
)
from astra_modulation_v2.dataset_builder import IQPreprocessorV2, iq_to_tensor_1d, iq_to_spectrogram_2d
from astra_modulation_v2.models.resnet1d import ResNet1DV2
from astra_modulation_v2.models.spectrogram_cnn import SpectrogramCNN2DV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TRAIN ENGINE] Using device: {DEVICE}")


class ModulationAugmentedDataset(Dataset):
    """Zero-leakage in-memory dataset providing both 1D and 2D features."""
    
    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        window_size: int = 2048,
        windows_per_capture: int = 4,
    ):
        self.window_size = window_size
        self.preprocessor = IQPreprocessorV2(remove_dc=True, normalize_rms=True)
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        captures = [c for c in data["captures"] if c["split"] == split]
        self.samples_1d = []
        self.samples_2d = []
        self.labels = []
        self.meta_list = []
        
        print(f"Loading {len(captures)} {split} source captures...")
        for cap in captures:
            raw = np.fromfile(cap["iq_path"], dtype=np.float32)
            iq = raw[0::2] + 1j * raw[1::2]
            label_idx = CLASS_TO_IDX_V2[cap["modulation"]]
            
            # Extract non-overlapping windows
            n_avail = len(iq)
            for w_idx in range(windows_per_capture):
                start = w_idx * window_size
                end = start + window_size
                if end > n_avail:
                    break
                w_iq = iq[start:end]
                w_clean = self.preprocessor.process(w_iq)
                
                # 1D tensor [2, 2048]
                t1d = iq_to_tensor_1d(w_clean)
                # 2D tensor [1, 128, 128]
                t2d = iq_to_spectrogram_2d(w_clean)
                
                self.samples_1d.append(t1d)
                self.samples_2d.append(t2d)
                self.labels.append(label_idx)
                self.meta_list.append(cap)
                
        self.samples_1d = torch.stack(self.samples_1d)
        self.samples_2d = torch.stack(self.samples_2d)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"Loaded {len(self.labels)} total {split} windowed samples.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.samples_1d[idx], self.samples_2d[idx], self.labels[idx]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_type: str = "1d",
    epochs: int = 15,
    lr: float = 1e-3,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Trains a model with AdamW and Cosine Annealing schedule."""
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_f1 = 0.0
    best_state = None
    history = []

    print(f"\nTraining {model.__class__.__name__} ({feature_type.upper()}) for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for b_1d, b_2d, targets in train_loader:
            x = (b_1d if feature_type == "1d" else b_2d).to(DEVICE)
            y = targets.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)

        scheduler.step()
        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_preds, val_targets, val_top3_hits = [], [], 0
        with torch.no_grad():
            for b_1d, b_2d, targets in val_loader:
                x = (b_1d if feature_type == "1d" else b_2d).to(DEVICE)
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                
                # Top-1
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.numpy())

                # Top-3
                top3 = torch.topk(logits, k=3, dim=-1).indices.cpu().numpy()
                for i, t in enumerate(targets.numpy()):
                    if t in top3[i]:
                        val_top3_hits += 1

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_top3_acc = val_top3_hits / len(val_targets)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_top3": round(val_top3_acc, 4),
        })

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% Top-3: {val_top3_acc*100:.1f}% Macro-F1: {val_f1*100:.1f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save best checkpoint
    if checkpoint_path and best_state:
        checkpoint_dict = {
            "state_dict": best_state,
            "model_architecture": model.__class__.__name__,
            "classes": list(MODULATION_CLASSES_V2),
            "class_schema_version": CLASS_SCHEMA_VERSION,
            "num_classes": NUM_CLASSES_V2,
            "input_shape": [2, 2048] if feature_type == "1d" else [1, 128, 128],
            "metrics": {
                "best_val_macro_f1": float(best_val_f1),
                "val_top3_accuracy": float(val_top3_acc),
            },
        }
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"[SAVED] Best checkpoint to: {checkpoint_path} (Val F1: {best_val_f1*100:.1f}%)")

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return {"best_f1": best_val_f1, "history": history}


if __name__ == "__main__":
    manifest_path = ROOT / "datasets" / "ASTRA_MODULATION_V3_AUGMENTED" / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    train_ds = ModulationAugmentedDataset(manifest_path, split="train", window_size=2048, windows_per_capture=4)
    val_ds = ModulationAugmentedDataset(manifest_path, split="val", window_size=2048, windows_per_capture=4)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 1. Train ResNet-1D V2
    model_1d = ResNet1DV2(in_channels=2, num_classes=NUM_CLASSES_V2)
    ckpt_1d_path = ROOT / "checkpoints" / "astra_resnet1d_v2.pt"
    res_1d = train_model(model_1d, train_loader, val_loader, feature_type="1d", epochs=15, lr=1e-3, checkpoint_path=ckpt_1d_path)

    # 2. Train Spectrogram CNN-2D V2
    model_2d = SpectrogramCNN2DV2(in_channels=1, num_classes=NUM_CLASSES_V2)
    ckpt_2d_path = ROOT / "checkpoints" / "astra_spectrogram_cnn_v2.pt"
    res_2d = train_model(model_2d, train_loader, val_loader, feature_type="2d", epochs=15, lr=1e-3, checkpoint_path=ckpt_2d_path)

    print("\n[TRAINING COMPLETE] Both ResNet-1D and CNN-2D models hardened on channel impairments.")
