"""
ResNet-1D Modulation Classifier Architecture (V2 Clean Rebuild).

Processes raw normalized complex IQ input tensor [B, 2, N] (N=2048).
Learns temporal phase transitions, instantaneous frequency evolution,
amplitude dynamics, and pulse shaping characteristics.

Features:
- Conv1D Stem with BatchNorm and GELU/ReLU
- 4 Residual Stages with multi-scale kernels (k=7 and k=5) and residual downsampling
- Dilated convolution support in middle stages for long-range symbol context
- Global AdaptiveAvgPool1D
- 256-dim bottleneck projection / embedding layer
- Linear classification head returning raw unnormalized logits for exactly 11 classes.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from astra_modulation_v2.class_schema import NUM_CLASSES_V2, MODULATION_CLASSES_V2


class Conv1DBlock(nn.Module):
    """Basic Conv1D - BatchNorm - Activation block with dilation support."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dilation: int = 1,
        activation: str = "gelu",
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Residual1DBlock(nn.Module):
    """Residual block with multi-scale kernel and shortcut connection."""
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class ResNet1DV2(nn.Module):
    """
    ResNet-1D Modulation Classifier V2.
    Clean, independently validated architecture for ASTRA.
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = NUM_CLASSES_V2,
        base_channels: int = 48,
        embedding_dim: int = 256,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.classes = list(MODULATION_CLASSES_V2)

        # 1. Stem: Large receptive field to capture symbol transitions
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 1: base_channels (48) -> 64
        c1 = base_channels
        c2 = base_channels * 2  # 96
        c3 = base_channels * 3  # 144
        c4 = base_channels * 4  # 192

        # 2. Stage 1
        self.stage1_down = Conv1DBlock(c1, c1, kernel_size=5, stride=1)
        self.stage1_res = Residual1DBlock(c1, kernel_size=7, dilation=1, dropout=dropout * 0.5)

        # 3. Stage 2 (downsampling)
        self.stage2_down = nn.Sequential(
            Conv1DBlock(c1, c2, kernel_size=5, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage2_res = Residual1DBlock(c2, kernel_size=5, dilation=2, dropout=dropout)

        # 4. Stage 3 (downsampling + dilation)
        self.stage3_down = nn.Sequential(
            Conv1DBlock(c2, c3, kernel_size=5, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage3_res = Residual1DBlock(c3, kernel_size=5, dilation=2, dropout=dropout)

        # 5. Stage 4 (final high-level representation)
        self.stage4_down = nn.Sequential(
            Conv1DBlock(c3, c4, kernel_size=5, stride=2),
        )
        self.stage4_res = Residual1DBlock(c4, kernel_size=5, dilation=1, dropout=dropout)

        # 6. Global Pooling & Embedding Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.bottleneck = nn.Sequential(
            nn.Linear(c4, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 7. Final Classification Head (no softmax)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input tensor [B, 2, N]
            return_embedding: If True, returns (logits, embedding)
        Returns:
            logits: [B, num_classes] (raw unnormalized logits)
            embedding: [B, embedding_dim] (optional)
        """
        out = self.stem(x)
        
        out = self.stage1_down(out)
        out = self.stage1_res(out)
        
        out = self.stage2_down(out)
        out = self.stage2_res(out)
        
        out = self.stage3_down(out)
        out = self.stage3_res(out)
        
        out = self.stage4_down(out)
        out = self.stage4_res(out)
        
        pooled = self.global_pool(out).flatten(1)  # [B, c4]
        embedding = self.bottleneck(pooled)         # [B, embedding_dim]
        logits = self.classifier(embedding)         # [B, num_classes]

        if return_embedding:
            return logits, embedding
        return logits, None
