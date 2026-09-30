"""
Checkpoint Management for ASTRA 2D Spectrogram Classifier.

Ensures strict reproducibility, class order validation, and configuration persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger("ASTRA_CHECKPOINT")


def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    class_names: Optional[List[str]] = None,
    model_config: Optional[Dict] = None,
    preprocessing_config: Optional[Dict] = None,
    spectrogram_config: Optional[Dict] = None,
    window_size: int = 2048,
    dataset_name: str = "CSPB.ML.2018R2",
    split_hash: Optional[str] = None,
    model_version: str = "astra_spectrogram_cnn_v1",
) -> None:
    """Saves a fully specified training checkpoint."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_metric": float(best_metric),
        "class_names": class_names or getattr(model, "class_names", []),
        "class_to_idx": {c: i for i, c in enumerate(class_names or getattr(model, "class_names", []))},
        "model_config": model_config or {},
        "preprocessing_config": preprocessing_config or {},
        "spectrogram_config": spectrogram_config or {},
        "window_size": window_size,
        "dataset_name": dataset_name,
        "split_hash": split_hash,
        "model_version": model_version,
    }
    torch.save(ckpt, p)
    logger.info(f"Saved checkpoint to {p} (Epoch {epoch}, Best Metric: {best_metric:.4f})")


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    device: Optional[torch.device] = None,
    strict_class_check: bool = True,
) -> Dict[str, Any]:
    """
    Loads checkpoint into model with class compatibility validation.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    device = device or torch.device("cpu")
    ckpt = torch.load(p, map_location=device, weights_only=False)

    ckpt_classes = ckpt.get("class_names", [])
    model_classes = getattr(model, "class_names", [])

    if strict_class_check and model_classes and ckpt_classes:
        if len(model_classes) != len(ckpt_classes):
            raise ValueError(
                f"Class count mismatch: Model has {len(model_classes)} classes, checkpoint has {len(ckpt_classes)} classes."
            )
        if [c.lower() for c in model_classes] != [c.lower() for c in ckpt_classes]:
            raise ValueError(
                f"Class order mismatch:\nModel: {model_classes}\nCheckpoint: {ckpt_classes}"
            )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    logger.info(f"Loaded checkpoint from {p} (Epoch {ckpt.get('epoch')}, Best Metric: {ckpt.get('best_metric', 0):.4f})")
    return ckpt
