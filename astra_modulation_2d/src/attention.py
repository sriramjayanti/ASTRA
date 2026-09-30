"""
Squeeze-and-Excitation (SE) Channel Attention Module for 2D Spectrogram Feature Maps.

Allows the network to adaptively recalibrate channel-wise feature responses by
explicitly modelling interdependencies between spectral/temporal channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEAttention2D(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Block for 2D CNNs.
    
    Architecture:
    Input [B, C, H, W]
    -> Global Average Pooling [B, C, 1, 1]
    -> Linear(C -> C // reduction, bias=False)
    -> ReLU / SiLU
    -> Linear(C // reduction -> C, bias=False)
    -> Sigmoid [B, C, 1, 1]
    -> Channel-wise product with original feature map
    """
    def __init__(self, channels: int, reduction: int = 16, activation: str = "relu"):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))

        act_layer = nn.SiLU(inplace=True) if activation.lower() == "silu" else nn.ReLU(inplace=True)

        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            act_layer,
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: [B, C, 1, 1] -> [B, C]
        squeezed = self.squeeze(x).view(b, c)
        # Excitation: [B, C] -> [B, C, 1, 1]
        weights = self.excitation(squeezed).view(b, c, 1, 1)
        # Scale
        return x * weights
