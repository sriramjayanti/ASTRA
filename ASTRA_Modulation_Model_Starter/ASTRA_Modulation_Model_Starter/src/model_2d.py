"""
ASTRA 2D Spectrogram CNN Modulation Classifier Architecture.

Performs Time-Frequency Spectral Analysis on Complex Baseband IQ Signals:
1. Differentiable / GPU STFT Spectrogram Extractor (Log-Power + Phase Derivative / Instantaneous Frequency)
2. Deep 2D ResNet Architecture with Residual Blocks, BatchNorm, Dropout, and Global Average Pooling
3. Outputs unnormalized class logits for 10-Class Modulation Classification
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramExtractor(nn.Module):
    """
    Extracts centered 2D Time-Frequency Spectrograms from 1D 2-Channel IQ tensors [B, 2, N].
    
    Channels:
    - Channel 0: Log-Power Spectral Density (PSD in dB, centered 0 Hz DC)
    - Channel 1: Instantaneous Frequency / Phase Derivative across time frames
    """
    def __init__(
        self,
        n_fft: int = 128,
        hop_length: int = 32,
        win_length: Optional[int] = 128,
        include_phase_channel: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.include_phase_channel = include_phase_channel
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Real tensor of shape [B, 2, N] (channel 0 = I, channel 1 = Q)
            
        Returns:
            spectrogram: Tensor of shape [B, C_out, Freq_Bins, Time_Frames]
                         where C_out = 2 (if include_phase_channel else 1)
        """
        # Form complex tensor [B, N]
        complex_iq = torch.complex(x[:, 0, :], x[:, 1, :])  # [B, N]

        # Compute STFT: [B, Freq_Bins (n_fft // 2 + 1), Time_Frames]
        # For baseband IQ (analytic/complex), we use full n_fft bins via torch.fft.fftshift
        stft_out = torch.stft(
            complex_iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=False,  # Complex signal has negative and positive frequencies
            return_complex=True,
        )  # Shape: [B, n_fft, Time_Frames]

        # Shift DC (0 Hz) to the center along frequency axis
        stft_shifted = torch.fft.fftshift(stft_out, dim=1)

        # 1. Log-Power Spectrogram (dB scale)
        power_spec = torch.abs(stft_shifted) ** 2
        log_power = 10.0 * torch.log10(power_spec + 1e-8)  # [B, F, T]

        # Standardize per-sample across (F, T) to [0, 1] / zero-mean unit-variance
        mean = log_power.mean(dim=(-2, -1), keepdim=True)
        std = log_power.std(dim=(-2, -1), keepdim=True) + 1e-6
        norm_power = (log_power - mean) / std

        if not self.include_phase_channel:
            return norm_power.unsqueeze(1)  # [B, 1, F, T]

        # 2. Instantaneous Phase Difference / Frequency Velocity
        phase = torch.angle(stft_shifted)  # [B, F, T] in [-pi, +pi]
        d_phase = torch.diff(phase, dim=-1, prepend=phase[:, :, :1])
        # Wrap phase differences to [-pi, +pi]
        d_phase_wrapped = (d_phase + math.pi) % (2.0 * math.pi) - math.pi
        d_phase_norm = d_phase_wrapped / math.pi  # Normalize to [-1, +1]

        # Stack into 2-channel 2D representation: [B, 2, F, T]
        return torch.stack([norm_power, d_phase_norm], dim=1)


class ResBlock2D(nn.Module):
    """2D Residual Convolutional Block with Pre-Activation or Standard Residual."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + res
        return self.relu(out)


class ResNet2DModClassifier(nn.Module):
    """
    ASTRA 2D ResNet Spectrogram Classifier for Modulation Recognition.
    
    Processes raw IQ [B, 2, N] via on-the-fly STFT Spectrogram generation,
    then classifies using deep 2D Residual Convolutional networks.
    """
    def __init__(
        self,
        num_classes: int = 10,
        n_fft: int = 128,
        hop_length: int = 32,
        input_channels: int = 2,  # [log_power, phase_velocity]
        base_channels: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.spectrogram_extractor = SpectrogramExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            include_phase_channel=(input_channels == 2),
        )

        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halve resolution
        )

        # 4 Residual Stages
        # Stage 1: base_channels
        self.stage1 = nn.Sequential(
            ResBlock2D(base_channels, base_channels, stride=1, dropout=dropout),
            ResBlock2D(base_channels, base_channels, stride=1, dropout=dropout),
        )

        # Stage 2: base_channels * 2
        c2 = base_channels * 2
        self.stage2 = nn.Sequential(
            ResBlock2D(base_channels, c2, stride=2, dropout=dropout),
            ResBlock2D(c2, c2, stride=1, dropout=dropout),
        )

        # Stage 3: base_channels * 4
        c3 = base_channels * 4
        self.stage3 = nn.Sequential(
            ResBlock2D(c2, c3, stride=2, dropout=dropout),
            ResBlock2D(c3, c3, stride=1, dropout=dropout),
        )

        # Stage 4: base_channels * 8
        c4 = base_channels * 8
        self.stage4 = nn.Sequential(
            ResBlock2D(c3, c4, stride=2, dropout=dropout),
            ResBlock2D(c4, c4, stride=1, dropout=dropout),
        )

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(c4, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 256-dimensional feature representation from raw IQ [B, 2, N] or 2D spectrogram [B, C, F, T].
        """
        if x.dim() == 3:  # Raw IQ [B, 2, N]
            spec = self.spectrogram_extractor(x)
        else:
            spec = x

        h = self.stem(spec)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        pooled = self.global_pool(h).flatten(1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepting either:
        1. Raw 1D IQ tensor [B, 2, N]
        2. Precomputed 2D Spectrogram [B, 2, F, T]
        
        Returns unnormalized class logits [B, num_classes].
        """
        feat = self.extract_features(x)
        feat = self.dropout_layer(feat)
        logits = self.classifier(feat)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
