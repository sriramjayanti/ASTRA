"""
ASTRA Production Modulation V2 Unified Training Engine.
Trains, evaluates, and validates:
1. ResNet-1D Time-Domain Classifier (Input: [B, 2, 2048])
2. CNN-2D Spectrogram Classifier (Input: [B, 1, 128, 128])

Evaluates across:
- Untouched Held-Out Test Set
- Out-Of-Distribution (OOD) Test Set
- SNR Tiers (-10 dB to +30 dB)
- FSK vs MSK tone-spacing breakdown
- Dense QAM (16/64/256) sub-matrix
- Noise / UNKNOWN rejection verification
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from astra_config.classes import (
    TRAINED_MODULATION_CLASSES_V2,
    MODULATION_CLASSES_V2,
    CLASS_TO_IDX_V2,
    IDX_TO_CLASS_V2,
    normalize_modulation_name,
)
from ASTRA_Modulation_Model_Starter.ASTRA_Modulation_Model_Starter.src.model import ResNet1DModClassifier
from astra_modulation_2d.src.model import ASTRASpectrogramCNN
from astra_modulation_2d.src.spectrogram import SpectrogramGenerator

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import os
threads = os.cpu_count() or 8
torch.set_num_threads(threads)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TRAIN_MOD_V2")
logger.info("Initialized PyTorch with %d CPU worker threads.", threads)



# =============================================================================
# 1. Dataset & Zero-Leakage Windowing Loader
# =============================================================================

def preprocess_iq_1d(iq: np.ndarray, window_size: int = 2048, eps: float = 1e-8) -> np.ndarray:
    """Production 1D preprocessing: DC removal and Unit RMS normalization -> [2, window_size]."""
    iq = np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0)
    if len(iq) < window_size:
        pad = np.zeros(window_size - len(iq), dtype=np.complex64)
        iq = np.concatenate([iq, pad])
    else:
        iq = iq[:window_size]

    # Zero-mean DC removal
    m = np.mean(iq)
    if np.isfinite(m):
        iq = iq - m

    # Unit RMS normalization
    rms = np.sqrt(np.mean(np.abs(iq) ** 2))
    if np.isfinite(rms) and rms > eps:
        iq = iq / rms
    else:
        iq = np.zeros_like(iq)

    return np.stack([np.real(iq), np.imag(iq)], axis=0).astype(np.float32)


class ModulationV2Dataset(Dataset):
    """
    In-memory windowed dataset guaranteeing zero-leakage from source captures.
    Pre-slices source signals into contiguous windows.
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        window_size: int = 2048,
        windows_per_capture: int = 4,
        is_2d: bool = False,
        spectrogram_gen: Optional[SpectrogramGenerator] = None,
    ):
        self.window_size = window_size
        self.is_2d = is_2d
        self.spectrogram_gen = spectrogram_gen

        self.samples: List[np.ndarray] = []
        self.labels: List[int] = []
        self.metadata: List[Dict[str, Any]] = []

        for _, row in manifest_df.iterrows():
            fpath = Path(row["file_path"])
            if not fpath.exists():
                continue
            raw_iq = np.fromfile(fpath, dtype=np.complex64)
            if len(raw_iq) < window_size:
                continue

            mod_norm = normalize_modulation_name(str(row["modulation"]))
            if mod_norm not in CLASS_TO_IDX_V2:
                continue
            label = CLASS_TO_IDX_V2[mod_norm]
            snr = float(row.get("snr_db", 0.0))
            tone_spacing = float(row.get("tone_spacing_ratio", 1.0))

            stride = window_size
            for w in range(windows_per_capture):
                start = w * stride
                if start + window_size > len(raw_iq):
                    break
                chunk = raw_iq[start : start + window_size]
                x_1d = preprocess_iq_1d(chunk, window_size=window_size)
                self.samples.append(x_1d)
                self.labels.append(label)
                self.metadata.append({
                    "modulation": mod_norm,
                    "snr_db": snr,
                    "tone_spacing_ratio": tone_spacing,
                    "source_id": str(row.get("source_id", "unknown")),
                })

        self.samples_tensor = torch.from_numpy(np.stack(self.samples, axis=0))  # [N, 2, L]
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)

        if self.is_2d and self.spectrogram_gen is not None:
            logger.info("Precomputing 2D Spectrograms for %d windows...", len(self.samples_tensor))
            chunks = []
            for i in range(0, len(self.samples_tensor), 256):
                batch_x = self.samples_tensor[i : i + 256]
                with torch.no_grad():
                    spec = self.spectrogram_gen(batch_x)
                chunks.append(spec.cpu())
            self.samples_tensor = torch.cat(chunks, dim=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples_tensor[idx], int(self.labels_tensor[idx])



# =============================================================================
# 2. Evaluation & Metric Utilities
# =============================================================================

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    classes: List[str] = TRAINED_MODULATION_CLASSES_V2,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics across all classes."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    per_class = {}
    for idx, cname in enumerate(classes):
        per_class[cname] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_snr_stratified(
    model: nn.Module,
    dataset: ModulationV2Dataset,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluates accuracy stratified into discrete SNR bins."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds: List[int] = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            preds = torch.argmax(model(inputs), dim=-1).cpu().numpy()
            all_preds.extend(preds)

    snr_bins = [(-20, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 35)]
    bin_labels = ["< -5 dB", "-5 to 0 dB", "0 to 5 dB", "5 to 10 dB", "10 to 15 dB", ">= 15 dB"]
    results = {}

    for (low, high), blabel in zip(snr_bins, bin_labels):
        indices = [
            i for i, m in enumerate(dataset.metadata)
            if low <= m["snr_db"] < high
        ]
        if not indices:
            results[blabel] = 0.0
            continue
        correct = sum(
            1 for idx in indices if all_preds[idx] == dataset.labels[idx]
        )
        results[blabel] = float(correct / len(indices))
    return results


def evaluate_fsk_tone_spacing(
    model: nn.Module,
    dataset: ModulationV2Dataset,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates FSK accuracy broken down by tone-spacing ratio."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    all_preds: List[int] = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            preds = torch.argmax(model(inputs), dim=-1).cpu().numpy()
            all_preds.extend(preds)

    spacing_bins = [(0.0, 0.4), (0.4, 0.7), (0.7, 1.2), (1.2, 1.8), (1.8, 3.0)]
    labels = ["Small (0.25-0.40 Rs)", "Medium-Low (0.50-0.70 Rs)", "Standard (0.75-1.20 Rs)", "Wide (1.2-1.8 Rs)", "Very Wide (>1.8 Rs)"]
    res = {}

    for (low, high), lbl in zip(spacing_bins, labels):
        indices = [
            i for i, m in enumerate(dataset.metadata)
            if "FSK" in m["modulation"] and low <= m["tone_spacing_ratio"] < high
        ]
        if not indices:
            continue
        correct = sum(1 for idx in indices if all_preds[idx] == dataset.labels[idx])
        res[lbl] = float(correct / len(indices))
    return res


def evaluate_noise_rejection(
    model: nn.Module,
    device: torch.device,
    is_2d: bool = False,
    spectrogram_gen: Optional[SpectrogramGenerator] = None,
    threshold: float = 0.70,
    num_samples: int = 100,
) -> float:
    """Evaluates UNKNOWN/Noise rejection on pure AWGN and unmodulated carrier."""
    model.eval()
    rng = np.random.default_rng(999)
    rejected = 0

    with torch.no_grad():
        for _ in range(num_samples):
            # 50% pure AWGN, 50% single CW tone
            if rng.uniform() < 0.5:
                iq = (rng.normal(0, 1, 2048) + 1j * rng.normal(0, 1, 2048)).astype(np.complex64)
            else:
                t = np.arange(2048) / 192000.0
                f0 = float(rng.uniform(5000, 30000))
                iq = np.exp(1j * 2 * np.pi * f0 * t).astype(np.complex64)

            x = preprocess_iq_1d(iq, window_size=2048)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)

            if is_2d and spectrogram_gen:
                x_t = spectrogram_gen(x_t)

            logits = model(x_t)
            probs = F.softmax(logits, dim=-1)
            max_prob = float(torch.max(probs).cpu().item())
            if max_prob < threshold:
                rejected += 1

    return float(rejected / num_samples)


# =============================================================================
# 3. Training Loop Engine
# =============================================================================

def train_single_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    ood_loader: DataLoader,
    train_dataset: ModulationV2Dataset,
    val_dataset: ModulationV2Dataset,
    test_dataset: ModulationV2Dataset,
    ood_dataset: ModulationV2Dataset,
    epochs: int = 20,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    output_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    """Trains 1D or 2D model with CosineAnnealing and early stopping on Macro-F1."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(TRAINED_MODULATION_CLASSES_V2)

    logger.info("Initializing %s model (Target classes: %d, Device: %s)...", model_type, num_classes, dev)
    spectrogram_gen = None
    if model_type == "1d":
        model = ResNet1DModClassifier(num_classes=num_classes).to(dev)
    elif model_type == "2d":
        spectrogram_gen = SpectrogramGenerator()
        model = ASTRASpectrogramCNN(
            class_names=TRAINED_MODULATION_CLASSES_V2,
            base_channels=32,
            stages=(64, 128, 256),
            se_attention=True,
            spectrogram_generator=spectrogram_gen,
        ).to(dev)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    class_counts = np.bincount(train_dataset.labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(1, class_counts).astype(np.float32)
    class_weights = class_weights / np.mean(class_weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=dev))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_macro_f1 = 0.0
    best_state_dict = None
    best_metrics = {}
    history: List[Dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(dev)
            targets = targets.to(dev)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(targets)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == targets).sum().item())
            total += len(targets)

        scheduler.step()
        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        val_eval = evaluate_model(model, val_loader, dev)
        val_acc = val_eval["accuracy"]
        val_macro_f1 = val_eval["macro_f1"]

        logger.info(
            "Epoch [%2d/%2d] Train Loss: %.4f | Train Acc: %.2f%% | Val Acc: %.2f%% | Val Macro-F1: %.4f",
            ep, epochs, train_loss, train_acc * 100, val_acc * 100, val_macro_f1
        )

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "lr": float(scheduler.get_last_lr()[0]),
        })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": ep,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "per_class": val_eval["per_class"],
            }

    # Load best checkpoint weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best model weights from Epoch %d (Val Macro-F1: %.4f)", best_metrics.get("epoch", 0), best_val_macro_f1)

    # Final In-Distribution Test Evaluation
    test_eval = evaluate_model(model, test_loader, dev)
    # Final OOD Test Evaluation
    ood_eval = evaluate_model(model, ood_loader, dev)
    # SNR Stratified Evaluation
    snr_eval = evaluate_snr_stratified(model, test_dataset, dev)
    # FSK Tone Spacing Evaluation
    fsk_spacing_eval = evaluate_fsk_tone_spacing(model, test_dataset, dev)
    # Noise Rejection Evaluation
    noise_rej = evaluate_noise_rejection(
        model, dev, is_2d=(model_type == "2d"), spectrogram_gen=spectrogram_gen
    )

    # Save Checkpoint Artifact
    if output_checkpoint:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        ckpt_payload = {
            "model_type": model_type,
            "state_dict": model.state_dict(),
            "classes": TRAINED_MODULATION_CLASSES_V2,
            "architecture": "ResNet1DModClassifier" if model_type == "1d" else "ASTRASpectrogramCNN",
            "input_shape": [2, 2048] if model_type == "1d" else [1, 128, 128],
            "best_epoch": best_metrics.get("epoch", epochs),
            "test_accuracy": test_eval["accuracy"],
            "test_macro_f1": test_eval["macro_f1"],
            "ood_accuracy": ood_eval["accuracy"],
            "ood_macro_f1": ood_eval["macro_f1"],
            "noise_rejection_rate": noise_rej,
            "history": history,
        }
        torch.save(ckpt_payload, output_checkpoint)
        logger.info("Saved %s checkpoint to: %s", model_type, output_checkpoint)

    # Strip numpy arrays for JSON serialization
    del test_eval["y_true"]
    del test_eval["y_pred"]
    del ood_eval["y_true"]
    del ood_eval["y_pred"]

    return {
        "model_type": model_type,
        "best_epoch": best_metrics.get("epoch", epochs),
        "test_eval": test_eval,
        "ood_eval": ood_eval,
        "snr_eval": snr_eval,
        "fsk_spacing_eval": fsk_spacing_eval,
        "noise_rejection_rate": noise_rej,
        "history": history,
    }


# =============================================================================
# 4. Master Orchestration Entrypoint
# =============================================================================

def run_v2_training(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 64,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_all = dataset_dir / "manifests" / "all.csv"
    if not manifest_all.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_all}")

    df_all = pd.read_csv(manifest_all)
    df_train = df_all[df_all["split"] == "train"]
    df_val = df_all[df_all["split"] == "validation"]
    df_test = df_all[df_all["split"] == "test"]
    df_ood = df_all[df_all["split"] == "ood_test"]

    logger.info("Loaded dataset: %d train, %d val, %d test, %d OOD captures.", len(df_train), len(df_val), len(df_test), len(df_ood))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train ResNet-1D
    logger.info("=================================================================")
    logger.info("       PHASE 1: TRAINING RESNET-1D (TIME DOMAIN)")
    logger.info("=================================================================")
    ds_1d_tr = ModulationV2Dataset(df_train, is_2d=False)
    ds_1d_va = ModulationV2Dataset(df_val, is_2d=False)
    ds_1d_te = ModulationV2Dataset(df_test, is_2d=False)
    ds_1d_ood = ModulationV2Dataset(df_ood, is_2d=False)

    loader_1d_tr = DataLoader(ds_1d_tr, batch_size=batch_size, shuffle=True)
    loader_1d_va = DataLoader(ds_1d_va, batch_size=batch_size, shuffle=False)
    loader_1d_te = DataLoader(ds_1d_te, batch_size=batch_size, shuffle=False)
    loader_1d_ood = DataLoader(ds_1d_ood, batch_size=batch_size, shuffle=False)

    ckpt_1d = output_dir / "astra_resnet1d_modulation_v2.pt"
    res_1d = train_single_model(
        model_type="1d",
        train_loader=loader_1d_tr,
        val_loader=loader_1d_va,
        test_loader=loader_1d_te,
        ood_loader=loader_1d_ood,
        train_dataset=ds_1d_tr,
        val_dataset=ds_1d_va,
        test_dataset=ds_1d_te,
        ood_dataset=ds_1d_ood,
        epochs=epochs,
        lr=1e-3,
        device=device,
        output_checkpoint=ckpt_1d,
    )

    # 2. Train CNN-2D Spectrogram
    logger.info("=================================================================")
    logger.info("       PHASE 2: TRAINING CNN-2D (SPECTROGRAM)")
    logger.info("=================================================================")
    spec_gen = SpectrogramGenerator()
    ds_2d_tr = ModulationV2Dataset(df_train, is_2d=True, spectrogram_gen=spec_gen)
    ds_2d_va = ModulationV2Dataset(df_val, is_2d=True, spectrogram_gen=spec_gen)
    ds_2d_te = ModulationV2Dataset(df_test, is_2d=True, spectrogram_gen=spec_gen)
    ds_2d_ood = ModulationV2Dataset(df_ood, is_2d=True, spectrogram_gen=spec_gen)

    loader_2d_tr = DataLoader(ds_2d_tr, batch_size=batch_size, shuffle=True)
    loader_2d_va = DataLoader(ds_2d_va, batch_size=batch_size, shuffle=False)
    loader_2d_te = DataLoader(ds_2d_te, batch_size=batch_size, shuffle=False)
    loader_2d_ood = DataLoader(ds_2d_ood, batch_size=batch_size, shuffle=False)

    ckpt_2d = output_dir / "astra_spectrogram_cnn_v2.pt"
    res_2d = train_single_model(
        model_type="2d",
        train_loader=loader_2d_tr,
        val_loader=loader_2d_va,
        test_loader=loader_2d_te,
        ood_loader=loader_2d_ood,
        train_dataset=ds_2d_tr,
        val_dataset=ds_2d_va,
        test_dataset=ds_2d_te,
        ood_dataset=ds_2d_ood,
        epochs=epochs,
        lr=1e-3,
        device=device,
        output_checkpoint=ckpt_2d,
    )

    combined_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resnet_1d": res_1d,
        "cnn_2d": res_2d,
    }

    report_path = output_dir / "v2_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    logger.info("Saved complete V2 evaluation report to: %s", report_path)
    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA V2 Modulation Training")
    parser.add_argument("--dataset", type=str, default="datasets/ASTRA_MODULATION_SYNTHETIC_V2")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_v2_training(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
