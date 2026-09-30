"""
ASTRA Mode B: Learned Feature & Logit Fusion MLP Module.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .confidence import ConfidenceCalculator
from .models import (
    BranchPrediction,
    CandidateItem,
    FusionPrediction,
)
from .validation import (
    validate_class_alignment,
    validate_source_alignment,
)


class LearnedFusionMLP(nn.Module):
    """
    Compact Multi-Layer Perceptron (MLP) for learned feature and logit fusion.
    
    Input vector:
      concat([feature_1d, feature_2d, logits_1d, logits_2d])
    
    Architecture:
      Input [B, D_in]
        ↓
      Linear(D_in → 256) → BatchNorm1D → ReLU/SiLU → Dropout(0.20)
        ↓
      Linear(256 → 128) → BatchNorm1D → ReLU/SiLU → Dropout(0.20)
        ↓
      Linear(128 → num_classes) → Raw Logits [B, num_classes]
    """

    def __init__(
        self,
        dim_1d_feature: int = 256,
        dim_2d_feature: int = 256,
        num_classes: int = 8,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.20,
        activation: str = "relu",
    ):
        super().__init__()
        self.dim_1d_feature = dim_1d_feature
        self.dim_2d_feature = dim_2d_feature
        self.num_classes = num_classes
        self.input_dim = dim_1d_feature + dim_2d_feature + (2 * num_classes)

        act_cls = nn.SiLU if activation.lower() == "silu" else nn.ReLU

        layers: List[nn.Module] = []
        prev_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(
        self,
        feat_1d: torch.Tensor,
        feat_2d: torch.Tensor,
        logits_1d: torch.Tensor,
        logits_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            Tuple[logits (B, num_classes), fused_embedding (B, hidden_dims[-1])]
        """
        x = torch.cat([feat_1d, feat_2d, logits_1d, logits_2d], dim=-1)
        fused_embedding = self.backbone(x)
        logits = self.classifier(fused_embedding)
        return logits, fused_embedding


class FusionFeatureDataset(Dataset):
    """
    Dataset containing pre-extracted features and logits from frozen 1D and 2D branches.
    """

    def __init__(
        self,
        feat_1d: np.ndarray,
        feat_2d: np.ndarray,
        logits_1d: np.ndarray,
        logits_2d: np.ndarray,
        labels: np.ndarray,
        source_ids: Optional[List[str]] = None,
    ):
        self.feat_1d = torch.tensor(feat_1d, dtype=torch.float32)
        self.feat_2d = torch.tensor(feat_2d, dtype=torch.float32)
        self.logits_1d = torch.tensor(logits_1d, dtype=torch.float32)
        self.logits_2d = torch.tensor(logits_2d, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.source_ids = source_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "feat_1d": self.feat_1d[idx],
            "feat_2d": self.feat_2d[idx],
            "logits_1d": self.logits_1d[idx],
            "logits_2d": self.logits_2d[idx],
            "label": self.labels[idx],
        }
        return item


class LearnedFusionEngine:
    """
    Inference and training coordinator for Mode B Learned Fusion.
    """

    def __init__(
        self,
        model: LearnedFusionMLP,
        class_names: List[str],
        top_k: int = 3,
        confidence_calc: Optional[ConfidenceCalculator] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.top_k = top_k
        self.confidence_calc = confidence_calc or ConfidenceCalculator()
        self.device = torch.device(device)
        self.model.eval()

    def fuse_predictions(
        self,
        pred_1d: BranchPrediction,
        pred_2d: BranchPrediction,
    ) -> FusionPrediction:
        """
        Fuses predictions using the trained learned MLP.
        """
        validate_class_alignment(pred_1d.class_names, pred_2d.class_names)
        validate_source_alignment(
            pred_1d.source_signal_id, pred_1d.window_start, pred_1d.window_end,
            pred_2d.source_signal_id, pred_2d.window_start, pred_2d.window_end,
        )

        if pred_1d.feature_embedding is None or pred_2d.feature_embedding is None:
            raise ValueError(
                "Learned fusion requires feature embeddings from both branches. "
                "Ensure feature_embedding is populated."
            )

        f1 = torch.tensor(pred_1d.feature_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
        f2 = torch.tensor(pred_2d.feature_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
        l1 = torch.tensor(pred_1d.logits, dtype=torch.float32, device=self.device).unsqueeze(0)
        l2 = torch.tensor(pred_2d.logits, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            logits, fused_emb = self.model(f1, f2, l1, l2)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Branch agreement
        pred_class_1d = pred_1d.predicted_class
        pred_class_2d = pred_2d.predicted_class
        branch_agreement = (pred_class_1d.strip().lower() == pred_class_2d.strip().lower())

        # Top-K ranking
        sorted_indices = np.argsort(probs)[::-1]
        best_idx = int(sorted_indices[0])
        fused_class = self.class_names[best_idx]
        fused_confidence = float(probs[best_idx])

        top_k_candidates: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indices[: self.top_k], start=1):
            top_k_candidates.append({
                "rank": rank,
                "class": self.class_names[idx],
                "probability": float(probs[idx]),
            })

        top1_p, top2_p, margin = self.confidence_calc.compute_margin(probs)
        entropy = self.confidence_calc.compute_entropy(probs)
        is_unknown = (fused_class.strip().lower() == "unknown")

        status, unknown_reason = self.confidence_calc.determine_status(
            predicted_class=fused_class,
            confidence=fused_confidence,
            confidence_margin=margin,
            branch_agreement=branch_agreement,
            entropy=entropy,
            is_unknown_class=is_unknown,
        )

        branch_evidence = {
            "resnet1d": {
                "model_name": pred_1d.model_name,
                "predicted_class": pred_1d.predicted_class,
                "confidence": float(pred_1d.confidence),
            },
            "spectrogram2d": {
                "model_name": pred_2d.model_name,
                "predicted_class": pred_2d.predicted_class,
                "confidence": float(pred_2d.confidence),
            },
        }

        probs_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        model_versions = {
            "resnet1d": pred_1d.model_version,
            "spectrogram2d": pred_2d.model_version,
            "fusion": "astra_learned_fusion_v1.0",
        }

        return FusionPrediction(
            predicted_class=fused_class,
            confidence=fused_confidence,
            status=status,
            unknown_reason=unknown_reason,
            branch_agreement=branch_agreement,
            confidence_margin=margin,
            probability_entropy=entropy,
            fusion_mode="learned",
            top_k=top_k_candidates,
            probabilities=probs_dict,
            branch_evidence=branch_evidence,
            model_versions=model_versions,
            logits=logits.squeeze(0).cpu().tolist(),
            fused_feature=fused_emb.squeeze(0).cpu().tolist(),
            source_signal_id=pred_1d.source_signal_id or pred_2d.source_signal_id,
            window_start=pred_1d.window_start,
            window_end=pred_1d.window_end,
        )
