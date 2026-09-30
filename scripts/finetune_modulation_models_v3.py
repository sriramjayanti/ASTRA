"""
finetune_modulation_models_v3.py
Fine-tunes base V2 checkpoints on the channel-augmented dataset with domain adaptation.
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
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import (
    CLASS_SCHEMA_VERSION,
    MODULATION_CLASSES_V2,
    NUM_CLASSES_V2,
)
from astra_modulation_v2.dataset_builder import IQPreprocessorV2, iq_to_tensor_1d, iq_to_spectrogram_2d
from astra_modulation_v2.models.resnet1d import ResNet1DV2
from astra_modulation_v2.models.spectrogram_cnn import SpectrogramCNN2DV2
from scripts.train_modulation_models_v3 import ModulationAugmentedDataset, DEVICE


def finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_type: str = "1d",
    epochs: int = 20,
    lr: float = 3e-4,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fine-tunes a pre-trained model with lower learning rate and cosine annealing."""
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    best_val_top3 = 0.0
    best_val_f1 = 0.0
    best_state = None

    print(f"\nFine-tuning {model.__class__.__name__} ({feature_type.upper()}) for {epochs} epochs at lr={lr}...")
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
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.numpy())

                top3 = torch.topk(logits, k=3, dim=-1).indices.cpu().numpy()
                for i, t in enumerate(targets.numpy()):
                    if t in top3[i]:
                        val_top3_hits += 1

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_top3_acc = val_top3_hits / len(val_targets)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% Top-3: {val_top3_acc*100:.1f}% Macro-F1: {val_f1*100:.1f}%")

        if val_top3_acc > best_val_top3 or (val_top3_acc == best_val_top3 and val_f1 > best_val_f1):
            best_val_top3 = val_top3_acc
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

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
                "val_top3_accuracy": float(best_val_top3),
            },
        }
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"[SAVED] Best checkpoint to: {checkpoint_path} (Val Top-3: {best_val_top3*100:.1f}%, F1: {best_val_f1*100:.1f}%)")

    return {"best_top3": best_val_top3, "best_f1": best_val_f1}


if __name__ == "__main__":
    manifest_path = ROOT / "datasets" / "ASTRA_MODULATION_V3_AUGMENTED" / "manifest.json"
    train_ds = ModulationAugmentedDataset(manifest_path, split="train", window_size=2048, windows_per_capture=4)
    val_ds = ModulationAugmentedDataset(manifest_path, split="val", window_size=2048, windows_per_capture=4)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 1. Fine-tune ResNet-1D
    model_1d = ResNet1DV2(in_channels=2, num_classes=NUM_CLASSES_V2)
    ckpt_1d_path = ROOT / "checkpoints" / "astra_resnet1d_v2.pt"
    if ckpt_1d_path.exists():
        raw_ckpt = torch.load(ckpt_1d_path, map_location="cpu")
        state = raw_ckpt.get("state_dict", raw_ckpt)
        model_1d.load_state_dict(state)
        print("Loaded initial weights for ResNet-1D.")
    res_1d = finetune_model(model_1d, train_loader, val_loader, feature_type="1d", epochs=20, lr=3e-4, checkpoint_path=ckpt_1d_path)

    # 2. Fine-tune Spectrogram CNN-2D
    model_2d = SpectrogramCNN2DV2(in_channels=1, num_classes=NUM_CLASSES_V2)
    ckpt_2d_path = ROOT / "checkpoints" / "astra_spectrogram_cnn_v2.pt"
    if ckpt_2d_path.exists():
        raw_ckpt = torch.load(ckpt_2d_path, map_location="cpu")
        state = raw_ckpt.get("state_dict", raw_ckpt)
        model_2d.load_state_dict(state)
        print("Loaded initial weights for CNN-2D.")
    res_2d = finetune_model(model_2d, train_loader, val_loader, feature_type="2d", epochs=20, lr=3e-4, checkpoint_path=ckpt_2d_path)
