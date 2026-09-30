"""
heads.py
Classification and prediction heads for bit-level sequence labeling, boundary detection, and frame-level tagging.
"""

from typing import Optional
import torch
import torch.nn as nn


class SequenceClassificationHead(nn.Module):
    """
    Per-position MLP head for bit-level structure classification.
    Maps [B, L, D] -> [B, L, num_classes].
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 128, num_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Contextual sequence representations [B, L, d_model]
        Returns:
            Per-position raw unnormalized logits [B, L, num_classes]
        """
        return self.mlp(x)


class BoundaryPredictionHead(nn.Module):
    """
    Auxiliary per-position binary boundary transition head.
    Maps [B, L, D] -> [B, L, 1] logits (boundary presence).
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1) # [B, L]


class FrameClassificationHead(nn.Module):
    """
    Optional frame-level multi-label classification head using masked average pooling.
    Maps [B, L, D] -> [B, num_frame_classes].
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 128, num_frame_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_frame_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence tensor [B, L, d_model]
            padding_mask: Boolean tensor [B, L] where True = padded
        Returns:
            Frame-level logits [B, num_frame_classes]
        """
        if padding_mask is not None:
            # Invert mask: True for valid elements
            valid_mask = (~padding_mask).unsqueeze(-1).float() # [B, L, 1]
            sum_x = torch.sum(x * valid_mask, dim=1) # [B, d_model]
            counts = torch.clamp(valid_mask.sum(dim=1), min=1.0) # [B, 1]
            pooled = sum_x / counts
        else:
            pooled = torch.mean(x, dim=1)

        return self.mlp(pooled)
