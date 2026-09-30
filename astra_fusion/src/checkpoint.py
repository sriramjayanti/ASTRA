"""
ASTRA Fusion Checkpoint Serialization and Deserialization.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from .learned_fusion import LearnedFusionEngine, LearnedFusionMLP


def save_fusion_checkpoint(
    path: str,
    fusion_model: LearnedFusionMLP,
    class_names: List[str],
    branch_versions: Dict[str, str],
    config: Optional[Dict[str, Any]] = None,
    validation_metrics: Optional[Dict[str, Any]] = None,
    checkpoint_version: str = "1.0.0",
) -> None:
    """
    Saves complete learned fusion checkpoint bundle.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "checkpoint_version": checkpoint_version,
        "fusion_state_dict": fusion_model.state_dict(),
        "class_names": class_names,
        "branch_versions": branch_versions,
        "dim_1d_feature": fusion_model.dim_1d_feature,
        "dim_2d_feature": fusion_model.dim_2d_feature,
        "num_classes": fusion_model.num_classes,
        "config": config or {},
        "validation_metrics": validation_metrics or {},
    }
    torch.save(payload, path)


def load_fusion_checkpoint(
    path: str,
    device: str = "cpu",
) -> Tuple[LearnedFusionMLP, Dict[str, Any]]:
    """
    Loads learned fusion model from checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    dim_1d = ckpt.get("dim_1d_feature", 256)
    dim_2d = ckpt.get("dim_2d_feature", 256)
    num_classes = ckpt.get("num_classes", len(ckpt.get("class_names", [])))
    
    model = LearnedFusionMLP(
        dim_1d_feature=dim_1d,
        dim_2d_feature=dim_2d,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["fusion_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
