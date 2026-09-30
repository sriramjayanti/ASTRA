"""
ASTRA Production 2D Spectrogram CNN Modulation Classifier.

Architecture:
Input [B, 1, F, T] (Log-power standardized spectrogram)
  ↓
STEM: Conv2D(1 → 32, 3x3, stride 1, padding 1) + BatchNorm2D + SiLU/ReLU
  ↓
STAGE 1: 32 → 64 (2 Residual Blocks, first block stride 2)
  ↓
STAGE 2: 64 → 128 (2 Residual Blocks, first block stride 2)
  ↓
STAGE 3: 128 → 256 (2 Residual Blocks, first block stride 2)
  ↓
SE ATTENTION (Reduction ratio = 16)
  ↓
AdaptiveAvgPool2D(1, 1) -> 256-Dimensional Feature Embedding
  ↓
CLASSIFIER HEAD: Linear(256 → 128) + BatchNorm1D + Act + Dropout(0.25) + Linear(128 → num_classes)
  ↓
RAW LOGITS [B, num_classes] (Softmax applied ONLY during inference)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .spectrogram import SpectrogramGenerator
from .residual import ResidualBlock2D
from .attention import SEAttention2D


class ASTRASpectrogramCNN(nn.Module):
    """
    ASTRA 2D Spectrogram Convolutional Neural Network with Squeeze-and-Excitation Attention.
    """
    def __init__(
        self,
        class_names: List[str],
        base_channels: int = 32,
        stages: Tuple[int, ...] = (64, 128, 256),
        blocks_per_stage: Tuple[int, ...] = (2, 2, 2),
        se_attention: bool = True,
        se_reduction: int = 16,
        embedding_dim: int = 256,
        dropout: float = 0.25,
        activation: str = "relu",
        spectrogram_generator: Optional[SpectrogramGenerator] = None,
    ):
        super().__init__()
        self.class_names = [c.lower().strip() for c in class_names]
        self.num_classes = len(self.class_names)
        self.base_channels = base_channels
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.spectrogram_generator = spectrogram_generator or SpectrogramGenerator()

        act_fn = nn.SiLU if activation.lower() == "silu" else nn.ReLU

        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            act_fn(inplace=True),
        )

        # 2. Residual Stages
        # Stage 1: base_channels -> stages[0]
        c1 = stages[0]
        st1_blocks = [ResidualBlock2D(base_channels, c1, stride=2, activation=activation, dropout=0.0)]
        for _ in range(blocks_per_stage[0] - 1):
            st1_blocks.append(ResidualBlock2D(c1, c1, stride=1, activation=activation, dropout=0.0))
        self.stage1 = nn.Sequential(*st1_blocks)

        # Stage 2: stages[0] -> stages[1]
        c2 = stages[1]
        st2_blocks = [ResidualBlock2D(c1, c2, stride=2, activation=activation, dropout=0.0)]
        for _ in range(blocks_per_stage[1] - 1):
            st2_blocks.append(ResidualBlock2D(c2, c2, stride=1, activation=activation, dropout=0.0))
        self.stage2 = nn.Sequential(*st2_blocks)

        # Stage 3: stages[1] -> stages[2]
        c3 = stages[2]
        st3_blocks = [ResidualBlock2D(c2, c3, stride=2, activation=activation, dropout=0.0)]
        for _ in range(blocks_per_stage[2] - 1):
            st3_blocks.append(ResidualBlock2D(c3, c3, stride=1, activation=activation, dropout=0.0))
        self.stage3 = nn.Sequential(*st3_blocks)

        # 3. SE Attention
        if se_attention:
            self.attention = SEAttention2D(c3, reduction=se_reduction, activation=activation)
        else:
            self.attention = nn.Identity()

        # 4. Global Average Pooling -> 256-D Embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 5. Classifier Head
        self.head = nn.Sequential(
            nn.Linear(c3, 128, bias=False),
            nn.BatchNorm1d(128),
            act_fn(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 256-dimensional feature representation from either:
        - Spectrogram tensor [B, 1, F, T]
        - Raw complex IQ tensor [B, N] or real 2-channel tensor [B, 2, N]
        
        Returns:
            features: Tensor of shape [B, 256]
        """
        if x.dim() == 2 or (x.dim() == 3 and x.size(1) == 2):
            # Generate spectrogram on the fly
            x = self.spectrogram_generator(x)

        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected spectrogram tensor of shape [B, 1, F, T], got {x.shape}")

        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.attention(h)
        feat = self.global_pool(h).flatten(1)  # [B, 256]
        return feat

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass alias returning raw logits [B, num_classes]."""
        feat = self.extract_features(x)
        logits = self.head(feat)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized class logits.
        Softmax is explicitly NOT applied here.
        """
        return self.predict_logits(x)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
