"""
ASTRA Production Unified 10-Class 1D ResNet Modulation Classifier Training (Vectorized).

Combines:
1. Real CSPB Captures (from all 28 zip archives: BPSK, QPSK, 8PSK, DQPSK, MSK, 16QAM, 64QAM, 256QAM)
2. Synthetic Captures (2-FSK, 4-FSK, multi-CFO, multi-SNR, fading channels, multi-sample-rate)

Target 10-Class Vocabulary:
["2fsk", "4fsk", "bpsk", "qpsk", "8psk", "dqpsk", "msk", "16qam", "64qam", "256qam"]

Architecture: ResNet1DModClassifier (Stem -> 4 Residual Stages -> AdaptiveAvgPool -> Linear)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# Insert starter src to path
workspace_root = Path(__file__).resolve().parent.parent
starter_src = workspace_root / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "src"
if str(starter_src) not in sys.path:
    sys.path.insert(0, str(starter_src))

from model import ResNet1DModClassifier
from cspb_loader import CSPBDatasetCatalog, create_cspb_file_splits
from dataset import read_tim_file

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TRAIN_PRODUCTION_1D")

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


class SignalRecord:
    def __init__(
        self,
        source: str,
        modulation: str,
        snr_db: float,
        filepath: Optional[Path] = None,
        raw_complex: Optional[np.ndarray] = None,
    ):
        self.source = source
        self.modulation = modulation.lower().strip()
        self.snr_db = float(snr_db)
        self.filepath = filepath
        self.raw_complex = raw_complex


def clean_and_normalize_signal(iq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean DC removal and RMS power normalization to avoid numerical instability."""
    iq = np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.abs(iq)
    peak = np.max(mag) if len(mag) > 0 else 0.0
    if peak > 2.0:
        iq = iq / peak

    if len(iq) > 0:
        m = np.mean(iq)
        if np.isfinite(m):
            iq = iq - m
        rms = np.sqrt(np.mean(np.abs(iq) ** 2))
        if np.isfinite(rms) and rms > eps:
            iq = iq / rms
        else:
            iq = np.zeros_like(iq)
    return np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)


def build_tensor_dataset(
    records: List[SignalRecord],
    classes: List[str] = UNIFIED_CLASSES,
    window_size: int = 2048,
    stride: int = 2048,
    windows_per_signal: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-extracts all IQ slices into contiguous Torch tensors for maximum CPU throughput."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    valid_recs = [r for r in records if r.modulation in class_to_idx and r.raw_complex is not None and len(r.raw_complex) > 0]

    total_windows = len(valid_recs) * windows_per_signal
    X_arr = np.zeros((total_windows, 2, window_size), dtype=np.float32)
    y_arr = np.zeros(total_windows, dtype=np.int64)

    idx = 0
    for rec in valid_recs:
        target = class_to_idx[rec.modulation]
        sig = clean_and_normalize_signal(rec.raw_complex)
        sig_len = len(sig)

        for w in range(windows_per_signal):
            start = w * stride
            if sig_len >= start + window_size:
                win = sig[start : start + window_size]
            else:
                win = np.zeros(window_size, dtype=np.complex64)
                avail = min(sig_len, window_size)
                win[:avail] = sig[:avail]

            # Store channel 0: Real (I), channel 1: Imag (Q)
            X_arr[idx, 0, :] = np.real(win).astype(np.float32)
            X_arr[idx, 1, :] = np.imag(win).astype(np.float32)
            y_arr[idx] = target
            idx += 1

    logger.info(f"Built contiguous tensor: {X_arr.shape} ({X_arr.nbytes / (1024*1024):.1f} MB in RAM)")
    return torch.from_numpy(X_arr[:idx]), torch.from_numpy(y_arr[:idx])


def load_all_synthetic_records(synthetic_dirs: List[Path]) -> Tuple[List[SignalRecord], List[SignalRecord], List[SignalRecord]]:
    train_recs, val_recs, test_recs = [], [], []

    for syn_dir in synthetic_dirs:
        manifest_all = syn_dir / "manifests" / "all.csv"
        if not manifest_all.exists():
            continue

        df = pd.read_csv(manifest_all)
        all_dir_recs = []
        for _, row in df.iterrows():
            mod = str(row["modulation"]).lower().strip()
            snr = float(row.get("inband_snr_db", row.get("snr_db", 10.0)))
            cap_col = "capture_path" if "capture_path" in row else "capture_file_path"
            cap_path_str = str(row[cap_col])
            full_cap_path = Path(cap_path_str)
            if not full_cap_path.is_absolute():
                full_cap_path = syn_dir / "captures" / Path(cap_path_str).name
            if not full_cap_path.exists():
                full_cap_path = syn_dir / "captures" / f"{Path(cap_path_str).stem}.iq"
            if not full_cap_path.exists():
                continue

            split_name = str(row.get("split", "train")).lower()
            rec = SignalRecord(
                source=f"syn_{syn_dir.name}",
                modulation=mod,
                snr_db=snr,
                filepath=full_cap_path,
            )
            all_dir_recs.append((rec, split_name))

        def _load_sig(r_tuple):
            r, s_name = r_tuple
            try:
                r.raw_complex = read_tim_file(r.filepath)
                return r, s_name
            except Exception:
                return None, s_name

        with ThreadPoolExecutor(max_workers=8) as ex:
            preloaded = list(ex.map(_load_sig, all_dir_recs))

        dir_train, dir_val, dir_test = 0, 0, 0
        for r, s_name in preloaded:
            if r is None or r.raw_complex is None or len(r.raw_complex) == 0:
                continue
            if s_name == "train":
                train_recs.append(r)
                dir_train += 1
            elif s_name in ("validation", "val"):
                val_recs.append(r)
                dir_val += 1
            else:
                test_recs.append(r)
                dir_test += 1

        logger.info(f"Loaded & preloaded synthetic dataset '{syn_dir.name}': {dir_train} train, {dir_val} val, {dir_test} test.")

    return train_recs, val_recs, test_recs


def load_cspb_records(
    data_dir: Path,
    max_per_class: int = 250,
) -> Tuple[List[SignalRecord], List[SignalRecord], List[SignalRecord]]:
    catalog = CSPBDatasetCatalog(data_dir)
    cspb_train_df, cspb_val_df, cspb_test_df = create_cspb_file_splits(
        catalog=catalog,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        max_signals_per_class=max_per_class,
    )

    def extract_from_df(df: pd.DataFrame) -> List[SignalRecord]:
        recs = []
        grouped = df.groupby("zip_path")
        for zip_p, group in grouped:
            with zipfile.ZipFile(zip_p, "r") as zf:
                for _, row in group.iterrows():
                    raw_b = zf.read(row["internal_path"])
                    arr = np.frombuffer(raw_b, dtype=np.float32)
                    i_samples = arr[0::2]
                    q_samples = arr[1::2]
                    complex_sig = (i_samples + 1j * q_samples).astype(np.complex64)
                    recs.append(SignalRecord(
                        source="cspb",
                        modulation=row["modulation"],
                        snr_db=row["snr_db"],
                        raw_complex=complex_sig,
                    ))
        return recs

    logger.info(f"Extracting CSPB partitions from 28 archives (max={max_per_class}/class)...")
    train_recs = extract_from_df(cspb_train_df)
    val_recs = extract_from_df(cspb_val_df)
    test_recs = extract_from_df(cspb_test_df)
    logger.info(f"Loaded CSPB partitions: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test.")
    return train_recs, val_recs, test_recs


def run_production_training(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_cspb_per_class: int = 250,
    windows_per_signal: int = 1,
):
    print("=" * 80, flush=True)
    print("ASTRA: PRODUCTION 10-CLASS 1D RESNET TRAINING (CSPB + THOUSANDS SYNTHETIC)", flush=True)
    print("=" * 80, flush=True)

    # Use all CPU cores
    torch.set_num_threads(8)

    cspb_dir = starter_src.parent / "data"
    out_dir = starter_src.parent / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_dirs = [
        workspace_root / "datasets" / "astra_synthetic_enriched_v2",
        workspace_root / "datasets" / "astra_fsk_enriched_v1",
        workspace_root / "datasets" / "astra_training_dataset",
        workspace_root / "datasets" / "astra_synthetic_v1",
    ]

    # 1. Load CSPB
    cspb_train, cspb_val, cspb_test = load_cspb_records(cspb_dir, max_per_class=max_cspb_per_class)

    # 2. Load Synthetic
    syn_train, syn_val, syn_test = load_all_synthetic_records(synthetic_dirs)

    train_recs = cspb_train + syn_train
    val_recs = cspb_val + syn_val
    test_recs = cspb_test + syn_test

    print(f"\n[DATASET SUMMARY]", flush=True)
    print(f"  • Training Signals:   {len(train_recs):,} (CSPB: {len(cspb_train)}, Synthetic: {len(syn_train)})", flush=True)
    print(f"  • Validation Signals: {len(val_recs):,} (CSPB: {len(cspb_val)}, Synthetic: {len(syn_val)})", flush=True)
    print(f"  • Testing Signals:    {len(test_recs):,} (CSPB: {len(cspb_test)}, Synthetic: {len(syn_test)})", flush=True)

    # Build Contiguous Vectorized Tensors in RAM
    print("\n[VECTORIZING IQ WINDOWS INTO RAM...]", flush=True)
    X_train, y_train = build_tensor_dataset(train_recs, classes=UNIFIED_CLASSES, windows_per_signal=windows_per_signal)
    X_val, y_val = build_tensor_dataset(val_recs, classes=UNIFIED_CLASSES, windows_per_signal=windows_per_signal)
    X_test, y_test = build_tensor_dataset(test_recs, classes=UNIFIED_CLASSES, windows_per_signal=windows_per_signal)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE]: {device} (PyTorch threads: {torch.get_num_threads()})", flush=True)

    # ResNet1DModClassifier
    model = ResNet1DModClassifier(
        num_classes=len(UNIFIED_CLASSES),
        input_channels=2,
        base_channels=64,
        dropout=0.2,
    ).to(device)

    param_count = model.count_parameters()
    print(f"Model Parameters: {param_count:,} ({param_count * 4 / (1024*1024):.2f} MB)", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_ckpt_path = out_dir / "best_model_1d_10class_unified.pt"
    root_ckpt_path = workspace_root / "best_model_resnet1d.pt"

    print(f"\n--- Starting Production Training Loop ({epochs} Epochs) ---", flush=True)
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            # Fast Vectorized Batch Augmentation (Phase rotation & subtle jitter)
            theta = torch.rand(bx.size(0), 1, 1, device=device) * (2.0 * math.pi)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            i_rot = bx[:, 0:1, :] * cos_t - bx[:, 1:2, :] * sin_t
            q_rot = bx[:, 0:1, :] * sin_t + bx[:, 1:2, :] * cos_t
            bx_aug = torch.cat([i_rot, q_rot], dim=1)

            # 10% AWGN noise injection
            if random.random() > 0.5:
                bx_aug = bx_aug + torch.randn_like(bx_aug) * 0.03

            optimizer.zero_grad()
            logits = model(bx_aug)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(by.cpu().numpy())

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        dt = time.time() - t0

        msg = (
            f"Epoch {epoch:2d}/{epochs:2d} [{dt:4.1f}s] | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}%, Macro F1: {val_f1*100:5.2f}%"
        )
        print(msg, flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "classes": UNIFIED_CLASSES,
                "dataset": "CSPB_28_Archives + ASTRA_Synthetic_Thousands",
                "config": {
                    "num_classes": len(UNIFIED_CLASSES),
                    "input_channels": 2,
                    "base_channels": 64,
                    "dropout": 0.2,
                },
            }
            torch.save(ckpt_data, best_ckpt_path)
            torch.save(ckpt_data, root_ckpt_path)
            print(f"  --> [SAVED NEW BEST MODEL] {best_ckpt_path} (F1: {val_f1*100:.2f}%)", flush=True)

    total_time = time.time() - t_start
    print("\n" + "=" * 80, flush=True)
    print(f"TRAINING COMPLETE IN {total_time:.1f}s ({total_time/60:.2f} min). BEST VAL F1: {best_val_f1*100:.2f}%", flush=True)
    print("=" * 80, flush=True)

    # Final Test Evaluation
    print("\n--- Final Test Evaluation on Held-Out Test Signals ---", flush=True)
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_targets.extend(by.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    print(f"[TEST METRICS] Accuracy: {test_acc*100:.2f}% | Macro F1: {test_f1*100:.2f}%\n", flush=True)
    print(classification_report(test_targets, test_preds, target_names=[c.upper() for c in UNIFIED_CLASSES], zero_division=0), flush=True)

    # Evaluate on the 15 Unseen Benchmark Captures
    evaluate_15_benchmarks(model, device)


def evaluate_15_benchmarks(model: nn.Module, device: torch.device):
    print("\n" + "=" * 80, flush=True)
    print("EVALUATION ON 15 REAL/UNSEEN BENCHMARK SIGNALS (test_signals/)", flush=True)
    print("=" * 80, flush=True)

    test_dir = workspace_root / "test_signals"
    manifest_p = test_dir / "MANIFEST.json"
    if not manifest_p.exists():
        print("Manifest not found in test_signals.")
        return

    with open(manifest_p, "r") as f:
        manifest = json.load(f)

    signals_meta = manifest if isinstance(manifest, list) else manifest.get("signals", [])
    model.eval()
    correct = 0
    total = len(signals_meta)

    header = f"{'SIGNAL':<32} | {'TRUE MOD':<10} | {'PRED MOD':<10} | {'CONF':<8} | {'STATUS'}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for s_info in signals_meta:
        s_id = s_info.get("signal_id", s_info.get("filename", "unknown"))
        true_mod = s_info.get("modulation", "").lower().strip()
        data_file = test_dir / s_info.get("filename", "")

        # Read signal file
        try:
            sig = read_tim_file(data_file)
        except Exception as e:
            print(f"{s_id:<32} | {true_mod.upper():<10} | ERROR ({e})", flush=True)
            continue

        if len(sig) == 0:
            continue

        # Extract centered 2048-sample slice
        if len(sig) >= 2048:
            mid = len(sig) // 2
            start = max(0, mid - 1024)
            win = sig[start : start + 2048]
        else:
            win = np.zeros(2048, dtype=np.complex64)
            win[:len(sig)] = sig

        win = clean_and_normalize_signal(win)
        i_ch = np.real(win).astype(np.float32)
        q_ch = np.imag(win).astype(np.float32)
        x_tensor = torch.from_numpy(np.stack([i_ch, q_ch], axis=0)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_mod = UNIFIED_CLASSES[pred_idx]
            conf = float(probs[pred_idx])

        norm_true = true_mod.replace("-", "").replace(" ", "").lower()
        norm_pred = pred_mod.replace("-", "").replace(" ", "").lower()
        is_match = (norm_pred == norm_true)
        if is_match:
            correct += 1
            status = "MATCH (OK)"
        else:
            status = f"DIFF ({pred_mod.upper()})"

        print(f"{s_id:<32} | {true_mod.upper():<10} | {pred_mod.upper():<10} | {conf*100:5.1f}%  | {status}", flush=True)

    print("-" * len(header), flush=True)
    acc = (correct / total) * 100 if total > 0 else 0.0
    print(f"BENCHMARK SCORE: {correct}/{total} ({acc:.1f}% Match Rate across 15 real test signals)\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_cspb", type=int, default=150)
    parser.add_argument("--windows_per_signal", type=int, default=1)
    args = parser.parse_args()

    run_production_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_cspb_per_class=args.max_cspb,
        windows_per_signal=args.windows_per_signal,
    )
