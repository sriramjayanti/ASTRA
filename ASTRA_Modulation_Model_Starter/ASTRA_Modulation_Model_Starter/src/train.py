import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import yaml
from sklearn.metrics import f1_score, accuracy_score

from dataset import (
    load_config,
    load_truth_metadata,
    create_file_level_splits,
    get_dataloader,
    IQPreprocessor
)
from model import create_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ASTRA_TRAIN")


def set_seed(seed: int = 42):
    """Sets random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_y, _ in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "accuracy": acc, "macro_f1": macro_f1}


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_y, _ in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def train_model(
    config_path: str = "config.yaml",
    data_dir_override: Optional[str] = None,
    truth_file_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None
):
    # Load configuration from single authoritative source
    config = load_config(config_path)

    d_cfg = config.get("dataset", {})
    t_cfg = config.get("training", {})
    p_cfg = config.get("preprocessing", {})
    m_cfg = config.get("model", {})
    classes = config["classes"]

    # Read configuration values
    tim_dtype = d_cfg.get("tim_dtype", "float32")
    data_dir = data_dir_override or d_cfg.get("data_dir", "data")
    truth_file = truth_file_override or d_cfg.get("truth_file", "data/truth.csv")
    output_dir = output_dir_override or t_cfg.get("output_dir", "checkpoints")

    seed = d_cfg.get("split_seed", 42)
    set_seed(seed)

    window_size = d_cfg.get("window_size", 2048)
    train_ratio = d_cfg.get("train_ratio", 0.70)
    val_ratio = d_cfg.get("val_ratio", 0.15)
    test_ratio = d_cfg.get("test_ratio", 0.15)
    
    batch_size = batch_size_override or t_cfg.get("batch_size", 64)
    epochs = epochs_override or t_cfg.get("epochs", 40)
    lr = lr_override or t_cfg.get("learning_rate", 1e-3)
    weight_decay = t_cfg.get("weight_decay", 1e-4)

    # Centralized Preprocessor
    preprocessor = IQPreprocessor(
        remove_dc=p_cfg.get("remove_dc", True),
        normalization=p_cfg.get("normalization", "rms"),
        epsilon=p_cfg.get("epsilon", 1e-8)
    )

    out_path = Path(output_dir)
    manifest_path = out_path / "manifests"
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Truth Metadata & Create File-Level Splits
    truth_path = Path(truth_file)
    logger.info(f"Loading Truth Metadata from: {truth_path}")
    truth_df = load_truth_metadata(truth_path)

    train_df, val_df, test_df = create_file_level_splits(
        data_dir=data_dir,
        truth_df=truth_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        output_dir=manifest_path,
        stratify_by_mod=True
    )
    logger.info(f"Signal Files: {len(train_df)} Train | {len(val_df)} Val | {len(test_df)} Test")

    # 2. Build DataLoaders
    train_loader = get_dataloader(
        manifest_df=train_df,
        data_dir=data_dir,
        tim_dtype=tim_dtype,
        classes=classes,
        batch_size=batch_size,
        shuffle=True,
        window_size=window_size,
        preprocessor=preprocessor
    )

    val_loader = get_dataloader(
        manifest_df=val_df,
        data_dir=data_dir,
        tim_dtype=tim_dtype,
        classes=classes,
        batch_size=batch_size,
        shuffle=False,
        window_size=window_size,
        preprocessor=preprocessor
    )

    if len(train_loader.dataset) == 0:
        raise ValueError("Train dataset is empty! Please verify dataset directory and signal files.")

    logger.info(f"Total Windows: {len(train_loader.dataset)} Train | {len(val_loader.dataset)} Val")

    # 3. Model, Loss, Optimizer, Scheduler
    model = create_model(config).to(device)
    logger.info(f"ResNet-1D Model initialized with {model.count_parameters():,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Training Loop
    best_macro_f1 = -1.0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        epoch_dur = time.time() - t_epoch_start

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']*100:.2f}% F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": current_lr
        }
        history.append(record)

        # Save Best Checkpoint
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_checkpoint_path = out_path / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_macro_f1": best_macro_f1,
                "config": config,
                "classes": classes,
                "seed": seed
            }, best_checkpoint_path)
            logger.info(f"  -> Saved new best model checkpoint (Val Macro-F1: {best_macro_f1:.4f})")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.2f} minutes. Best Val Macro-F1: {best_macro_f1:.4f}")

    # Save Final Checkpoint & History
    final_checkpoint_path = out_path / "final_model.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": config,
        "classes": classes,
        "seed": seed
    }, final_checkpoint_path)

    with open(out_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out_path / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)

    with open(out_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info(f"Artifacts successfully saved to {out_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA Model 1 Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory containing .tim files")
    parser.add_argument("--truth_file", type=str, default=None, help="Path to truth metadata file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for weights and logs")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        data_dir_override=args.data_dir,
        truth_file_override=args.truth_file,
        output_dir_override=args.output_dir,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr
    )
