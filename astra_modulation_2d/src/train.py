"""
ASTRA 2D Spectrogram Classifier Training Engine.

Supports:
- Config-driven training for both Mode A (CSPB) and Mode B (Synthetic 10-Class)
- Zero-leakage source-level split partition
- Dynamic class vocabulary configuration
- Validation Macro-F1 checkpoint selection
- Early stopping based on validation metrics
- Full training history persistence (JSON/CSV)
"""

from __future__ import annotations

import argparse
import json
import logging
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import sys
# Bootstrap sys.path for direct script execution
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from astra_modulation_2d.src.utils import set_seed, load_yaml_config, setup_logger
from astra_modulation_2d.src.preprocessing import IQPreprocessor
from astra_modulation_2d.src.spectrogram import SpectrogramGenerator
from astra_modulation_2d.src.model import ASTRASpectrogramCNN
from astra_modulation_2d.src.dataset import SignalWindowRecord, create_2d_dataloaders
from astra_modulation_2d.src.splits import verify_disjoint_splits, save_split_json, load_split_json
from astra_modulation_2d.src.checkpoint import save_checkpoint, load_checkpoint
from astra_modulation_2d.src.metrics import (
    compute_classification_metrics,
    analyze_confusion_pairs,
    plot_confusion_matrix,
    compute_accuracy_vs_snr,
    plot_accuracy_vs_snr,
)

logger = setup_logger("TRAIN_2D")


def build_cspb_records(
    data_dir: Path,
    class_names: List[str],
    split_file: Optional[Path] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_signals_per_class: Optional[int] = None,
    windows_per_signal: int = 4,
    window_size: int = 2048,
    stride: int = 2048,
) -> Tuple[List[SignalWindowRecord], List[SignalWindowRecord], List[SignalWindowRecord], str]:
    """Indexes and splits CSPB 28 batches at the signal file level."""
    truth_file = data_dir / "truth.txt"
    if not truth_file.exists():
        raise FileNotFoundError(f"truth.txt not found at: {truth_file}")

    truth_df = pd.read_csv(truth_file, sep=r"\s+", header=None, names=[
        "signal_index", "modulation", "t0", "carrier_offset", "rolloff", "u", "d", "snr_db", "noise_density_db"
    ])
    truth_df["modulation"] = truth_df["modulation"].astype(str).str.lower().str.strip()

    # Load corrupt signals blacklist if present
    bad_list_file = data_dir / "bad_signals_list.json"
    bad_signals = set()
    if bad_list_file.exists():
        with open(bad_list_file, "r") as f:
            bad_signals = set(json.load(f))

    import zipfile
    sig_map: Dict[int, Tuple[str, str]] = {}
    for b_idx in range(1, 29):
        zp = data_dir / f"CSPB.ML_.2018R2_{b_idx}.zip"
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".tim") and name not in bad_signals:
                    stem = Path(name).stem
                    sig_id = int(stem.replace("signal_", ""))
                    sig_map[sig_id] = (str(zp), name)

    class_to_idx = {c.lower(): i for i, c in enumerate(class_names)}
    valid_rows = []
    for _, row in truth_df.iterrows():
        sig_id = int(row["signal_index"])
        mod = str(row["modulation"]).lower()
        if sig_id in sig_map and mod in class_to_idx:
            rdict = row.to_dict()
            rdict["zip_path"], rdict["internal_path"] = sig_map[sig_id]
            valid_rows.append(rdict)

    df = pd.DataFrame(valid_rows)
    logger.info(f"Verified {len(df):,} matching signals in CSPB catalog.")

    # Splitting
    if split_file and Path(split_file).exists():
        train_ids, val_ids, test_ids = load_split_json(split_file)
        split_hash = "loaded_from_file"
    else:
        rng = np.random.default_rng(seed)
        train_ids, val_ids, test_ids = [], [], []
        for mod, group in df.groupby("modulation"):
            s_ids = group["signal_index"].unique()
            rng.shuffle(s_ids)
            if max_signals_per_class:
                s_ids = s_ids[:max_signals_per_class]
            n_tot = len(s_ids)
            n_tr = int(n_tot * train_ratio)
            n_v = int(n_tot * val_ratio)
            train_ids.extend(s_ids[:n_tr])
            val_ids.extend(s_ids[n_tr : n_tr + n_v])
            test_ids.extend(s_ids[n_tr + n_v :])

        split_hash = save_split_json(split_file or (data_dir / "cspb_splits.json"), train_ids, val_ids, test_ids)

    df_map = {int(r["signal_index"]): r for r in valid_rows}

    def make_records(id_list: List) -> List[SignalWindowRecord]:
        records = []
        for sid in id_list:
            sid_int = int(sid)
            if sid_int not in df_map:
                continue
            row = df_map[sid_int]
            mod = row["modulation"]
            t_idx = class_to_idx[mod]
            snr = float(row["snr_db"])
            zp = row["zip_path"]
            ip = row["internal_path"]

            for w in range(windows_per_signal):
                records.append(SignalWindowRecord(
                    source="cspb",
                    source_signal_id=str(sid_int),
                    modulation=mod,
                    target=t_idx,
                    snr_db=snr,
                    start_sample=w * stride,
                    window_size=window_size,
                    zip_path=zp,
                    internal_path=ip,
                ))
        return records

    train_recs = make_records(train_ids)
    val_recs = make_records(val_ids)
    test_recs = make_records(test_ids)

    return train_recs, val_recs, test_recs, split_hash


def build_synthetic_records(
    dataset_roots: List[Path],
    class_names: List[str],
    windows_per_signal: int = 4,
    window_size: int = 2048,
    stride: int = 2048,
) -> Tuple[List[SignalWindowRecord], List[SignalWindowRecord], List[SignalWindowRecord]]:
    """Loads synthetic signals across specified directories."""
    class_to_idx = {c.lower(): i for i, c in enumerate(class_names)}
    train_recs, val_recs, test_recs = [], [], []

    for root in dataset_roots:
        m_file = root / "manifests" / "all.csv"
        if not m_file.exists():
            continue
        df = pd.read_csv(m_file)
        for _, row in df.iterrows():
            mod = str(row["modulation"]).lower().strip()
            if mod not in class_to_idx:
                continue
            target = class_to_idx[mod]
            snr = float(row.get("inband_snr_db", row.get("snr_db", 10.0)))
            cap_col = "capture_path" if "capture_path" in row else "capture_file_path"
            cap_p = Path(str(row[cap_col]))
            if not cap_p.is_absolute():
                cap_p = Path.cwd() / cap_p
            if not cap_p.exists():
                cap_p = root / "captures" / cap_p.name
            if not cap_p.exists():
                continue

            sid = str(row.get("dataset_record_id", cap_p.stem))
            split_name = str(row.get("split", "train")).lower()

            for w in range(windows_per_signal):
                rec = SignalWindowRecord(
                    source="synthetic",
                    source_signal_id=sid,
                    modulation=mod,
                    target=target,
                    snr_db=snr,
                    start_sample=w * stride,
                    window_size=window_size,
                    filepath=cap_p,
                )
                if split_name == "train":
                    train_recs.append(rec)
                elif split_name in ("validation", "val"):
                    val_recs.append(rec)
                else:
                    test_recs.append(rec)

    return train_recs, val_recs, test_recs


def train_model(
    config_path: Union[str, Path],
    tiny_overfit: bool = False,
    override_epochs: Optional[int] = None,
) -> Dict:
    """Executes the complete 2D Spectrogram Classifier training pipeline."""
    cfg = load_yaml_config(config_path)
    ds_cfg = cfg.get("dataset", {})
    split_cfg = cfg.get("split", {})
    prep_cfg = cfg.get("preprocessing", {})
    spec_cfg = cfg.get("spectrogram", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    seed = int(split_cfg.get("seed", 42))
    set_seed(seed)

    class_names = [c.lower().strip() for c in ds_cfg.get("class_names", [])]
    window_size = int(ds_cfg.get("window_size", 2048))
    stride = int(ds_cfg.get("stride", 2048))
    windows_per_signal = int(ds_cfg.get("windows_per_signal", 4))

    preprocessor = IQPreprocessor(
        remove_dc=prep_cfg.get("dc_removal", True),
        rms_normalization=prep_cfg.get("rms_normalization", True),
        strict_finite=prep_cfg.get("strict_finite", True),
        epsilon=float(prep_cfg.get("epsilon", 1e-8)),
    )

    spec_gen = SpectrogramGenerator(
        n_fft=int(spec_cfg.get("n_fft", 128)),
        hop_length=int(spec_cfg.get("hop_length", 32)),
        win_length=int(spec_cfg.get("win_length", 128)),
        window=str(spec_cfg.get("window", "hann")),
        fftshift=bool(spec_cfg.get("fftshift", True)),
        representation=str(spec_cfg.get("representation", "log_power")),
        normalization=str(spec_cfg.get("normalization", "standard")),
        epsilon=float(spec_cfg.get("epsilon", 1e-8)),
    )

    def resolve_path(p_str: str) -> Path:
        p = Path(p_str)
        if p.is_absolute() and p.exists():
            return p
        if (Path.cwd() / p).exists():
            return Path.cwd() / p
        if (Path(config_path).parent / p).exists():
            return Path(config_path).parent / p
        repo_root = Path(__file__).resolve().parent.parent.parent
        if (repo_root / p).exists():
            return repo_root / p
        return Path.cwd() / p

    # 1. Load Data
    dtype = ds_cfg.get("type", "cspb")
    split_hash = None
    if dtype == "cspb":
        data_root = resolve_path(ds_cfg.get("root", "data"))
        split_f = ds_cfg.get("split_file")
        split_p = resolve_path(split_f) if split_f else None
        max_sig = train_cfg.get("max_signals_per_class", None)

        train_recs, val_recs, test_recs, split_hash = build_cspb_records(
            data_dir=data_root,
            class_names=class_names,
            split_file=split_p,
            train_ratio=float(split_cfg.get("train", 0.70)),
            val_ratio=float(split_cfg.get("validation", 0.15)),
            test_ratio=float(split_cfg.get("test", 0.15)),
            seed=seed,
            max_signals_per_class=max_sig,
            windows_per_signal=windows_per_signal,
            window_size=window_size,
            stride=stride,
        )
    else:
        # Synthetic / Combined
        main_root = resolve_path(ds_cfg.get("root"))
        roots = [main_root]
        for add_r in ds_cfg.get("additional_roots", []):
            roots.append(resolve_path(add_r))
        train_recs, val_recs, test_recs = build_synthetic_records(
            dataset_roots=roots,
            class_names=class_names,
            windows_per_signal=windows_per_signal,
            window_size=window_size,
            stride=stride,
        )

    if tiny_overfit:
        logger.info("=== RUNNING TINY OVERFIT SANITY CHECK (32 SAMPLES) ===")
        train_recs = train_recs[:32]
        val_recs = train_recs[:16]
        test_recs = train_recs[:16]
        override_epochs = override_epochs or 30

    batch_size = int(train_cfg.get("batch_size", 64))
    train_loader, val_loader, test_loader = create_2d_dataloaders(
        train_recs, val_recs, test_recs, class_names, batch_size=batch_size, preprocessor=preprocessor, spectrogram_generator=spec_gen
    )

    logger.info(f"Dataset partitions: {len(train_recs):,} train, {len(val_recs):,} val, {len(test_recs):,} test windows.")

    # 2. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTRASpectrogramCNN(
        class_names=class_names,
        base_channels=int(model_cfg.get("base_channels", 32)),
        stages=tuple(model_cfg.get("stages", [64, 128, 256])),
        blocks_per_stage=tuple(model_cfg.get("blocks_per_stage", [2, 2, 2])),
        se_attention=bool(model_cfg.get("se_attention", True)),
        se_reduction=int(model_cfg.get("se_reduction", 16)),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.25)),
        spectrogram_generator=spec_gen,
    ).to(device)

    logger.info(f"ASTRASpectrogramCNN initialized on {device}: {model.count_parameters():,} parameters.")

    criterion = nn.CrossEntropyLoss()
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = override_epochs or int(train_cfg.get("epochs", 50))
    patience = int(train_cfg.get("early_stopping_patience", 8))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_dir = Path("outputs") / ds_cfg.get("name", "experiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_model.pt"

    best_val_f1 = -1.0
    no_improve_epochs = 0
    history = []

    # 3. Training Loop
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        for bx, by, _, _, _ in train_loader:
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
        train_metrics = compute_classification_metrics(train_targets, train_preds, class_names)
        train_acc = train_metrics["overall_accuracy"]
        train_f1 = train_metrics["macro_f1"]

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by, _, _, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_metrics = compute_classification_metrics(val_targets, val_preds, class_names)
        val_acc = val_metrics["overall_accuracy"]
        val_f1 = val_metrics["macro_f1"]
        scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        dur = time.time() - t0
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({dur:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.1f}% F1: {val_f1:.4f} | LR: {cur_lr:.1e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "lr": cur_lr,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            save_checkpoint(
                filepath=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=val_f1,
                class_names=class_names,
                model_config=model_cfg,
                preprocessing_config=prep_cfg,
                spectrogram_config=spec_cfg,
                window_size=window_size,
                dataset_name=ds_cfg.get("name", "dataset"),
                split_hash=split_hash,
            )
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience and not tiny_overfit:
                logger.info(f"Early stopping triggered after {patience} epochs without validation F1 improvement.")
                break

    # Save History
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)

    return {
        "best_val_f1": best_val_f1,
        "checkpoint_path": str(best_ckpt_path),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA 2D Spectrogram Classifier Training")
    parser.add_argument("--config", type=str, default="configs/cspb.yaml", help="Path to config YAML")
    parser.add_argument("--tiny_overfit", action="store_true", help="Run tiny overfit sanity check")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()

    train_model(config_path=args.config, tiny_overfit=args.tiny_overfit, override_epochs=args.epochs)
