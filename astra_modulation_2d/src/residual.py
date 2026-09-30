"""
2D Residual Convolutional Blocks for ASTRA Spectrogram Classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    """
    Standard 2D Residual Block with Conv2D, BatchNorm2D, Activation, and 1x1 projection shortcut.
    
    Path:
    x -> Conv2D (3x3) -> BatchNorm2D -> Act -> Conv2D (3x3) -> BatchNorm2D -> (+) -> Act
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_fn = nn.SiLU if activation.lower() == "silu" else nn.ReLU

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_fn(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_fn(inplace=True)

        # Shortcut projection if spatial dimension or channel count changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.act2(out)
