"""
Spectrogram CNN-2D Modulation Classifier Architecture (V2 Clean Rebuild).

Processes centered log-power spectrogram tensor [B, 1, F, T] (128x128).
Learns spectral density distribution, harmonic tone states, FSK transitions,
carrier offset drift, and spectral rolloff patterns.

Features:
- Residual 2D Convolution stages
- Squeeze-and-Excitation (SE) channel attention
- AdaptiveAvgPool2D
- 256-dim bottleneck projection / embedding layer
- Linear classification head returning raw unnormalized logits for exactly 11 classes.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from astra_modulation_v2.class_schema import NUM_CLASSES_V2, MODULATION_CLASSES_V2


class SqueezeExcitation2D(nn.Module):
    """Channel Attention via Squeeze-and-Excitation."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class Residual2DBlock(nn.Module):
    """Residual 2D block with Squeeze-and-Excitation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        se_reduction: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation2D(out_channels, reduction=se_reduction)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act(out + res)
        return out


class SpectrogramCNN2DV2(nn.Module):
    """
    Spectrogram CNN-2D Modulation Classifier V2.
    Clean, independently validated architecture for ASTRA.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES_V2,
        base_channels: int = 32,
        embedding_dim: int = 256,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.classes = list(MODULATION_CLASSES_V2)

        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256

        # 2. Stage 1: [B, c1, 32, 32]
        self.stage1 = nn.Sequential(
            Residual2DBlock(c1, c1, stride=1, dropout=dropout * 0.5),
            Residual2DBlock(c1, c1, stride=1, dropout=dropout * 0.5),
        )

        # 3. Stage 2: [B, c2, 16, 16]
        self.stage2 = nn.Sequential(
            Residual2DBlock(c1, c2, stride=2, dropout=dropout),
            Residual2DBlock(c2, c2, stride=1, dropout=dropout),
        )

        # 4. Stage 3: [B, c3, 8, 8]
        self.stage3 = nn.Sequential(
            Residual2DBlock(c2, c3, stride=2, dropout=dropout),
            Residual2DBlock(c3, c3, stride=1, dropout=dropout),
        )

        # 5. Stage 4: [B, c4, 4, 4]
        self.stage4 = nn.Sequential(
            Residual2DBlock(c3, c4, stride=2, dropout=dropout),
            Residual2DBlock(c4, c4, stride=1, dropout=dropout),
        )

        # 6. Global Pooling & Embedding Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Sequential(
            nn.Linear(c4, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 7. Final Classification Head (no softmax)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
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
            x: Input tensor [B, 1, 128, 128]
            return_embedding: If True, returns (logits, embedding)
        Returns:
            logits: [B, num_classes] (raw unnormalized logits)
            embedding: [B, embedding_dim] (optional)
        """
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        
        pooled = self.global_pool(out).flatten(1)   # [B, c4]
        embedding = self.bottleneck(pooled)          # [B, embedding_dim]
        logits = self.classifier(embedding)          # [B, num_classes]

        if return_embedding:
            return logits, embedding
        return logits, None
