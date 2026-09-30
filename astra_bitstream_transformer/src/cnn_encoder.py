"""
cnn_encoder.py
1D CNN local motif feature extractor with residual blocks preserving exact bit resolution.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class Conv1DResidualBlock(nn.Module):
    """
    1D Residual Block with 2 Conv1D layers, BatchNorm, and SiLU activations.
    Preserves exact sequence length (stride=1, padding='same').
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out + res)
        return out


class BitstreamCNNEncoder(nn.Module):
    """
    1D CNN local motif feature extractor for multi-channel bitstream sequences.
    Preserves sequence resolution L (stride=1 throughout).
    Input shape:  [B, in_channels, L]
    Output shape: [B, out_channels, L]
    """

    def __init__(
        self,
        in_channels: int = 6,
        channels: Optional[List[int]] = None,
        kernel_size: int = 7,
        residual_blocks_per_stage: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        padding = kernel_size // 2
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.SiLU()
        )

        stages = []
        curr_in = channels[0]

        for stage_idx, stage_out in enumerate(channels):
            # First block handles channel transition if needed
            stages.append(Conv1DResidualBlock(curr_in, stage_out, kernel_size=kernel_size, dropout=dropout))
            # Subsequent blocks in the stage
            for _ in range(residual_blocks_per_stage - 1):
                stages.append(Conv1DResidualBlock(stage_out, stage_out, kernel_size=kernel_size, dropout=dropout))
            curr_in = stage_out

        self.stages = nn.Sequential(*stages)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, in_channels, L]
        Returns:
            Extracted local feature tensor [B, out_channels, L]
        """
        out = self.initial_conv(x)
        out = self.stages(out)
        return out
