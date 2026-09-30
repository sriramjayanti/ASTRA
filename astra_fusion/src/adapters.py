"""
ASTRA Model Adapters for 1D ResNet and 2D Spectrogram CNN.
Encapsulates preprocessing, feature extraction, and prediction formatting.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import TemperatureScaler
from .models import BranchPrediction
from astra_config.classes import TRAINED_MODULATION_CLASSES_V2


def preprocess_iq(
    x: Union[np.ndarray, torch.Tensor],
    target_length: int = 2048,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Standard IQ preconditioning:
    1. Handles complex64 [N] or real 2-channel [2, N]
    2. Zero-mean DC removal
    3. Unit RMS amplitude normalization
    4. Truncates/pads to target_length
    Returns: torch.Tensor [2, target_length] float32
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    # Sanitize NaNs and Infs
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=1.0, neginf=-1.0)

    # Convert complex64 [N] to real 2-channel [2, N]
    if np.iscomplexobj(x_np):
        if x_np.ndim == 1:
            i_comp = np.real(x_np)
            q_comp = np.imag(x_np)
            x_2ch = np.stack([i_comp, q_comp], axis=0)  # [2, N]
        elif x_np.ndim == 2:
            i_comp = np.real(x_np)
            q_comp = np.imag(x_np)
            x_2ch = np.stack([i_comp, q_comp], axis=1)  # [B, 2, N]
        else:
            raise ValueError(f"Unsupported complex array dimension: {x_np.shape}")
    else:
        if x_np.ndim == 1:
            x_2ch = np.stack([x_np, np.zeros_like(x_np)], axis=0)
        elif x_np.ndim == 2:
            if x_np.shape[0] == 2:
                x_2ch = x_np  # [2, N]
            elif x_np.shape[1] == 2:
                x_2ch = x_np.T  # [2, N]
            else:
                x_2ch = np.stack([x_np[0], x_np[1] if x_np.shape[0] > 1 else np.zeros_like(x_np[0])], axis=0)
        elif x_np.ndim == 3 and x_np.shape[1] == 2:
            x_2ch = x_np  # [B, 2, N]
        else:
            raise ValueError(f"Unsupported real array dimension: {x_np.shape}")

    # Process single sample [2, N] or batch [B, 2, N]
    if x_2ch.ndim == 2:
        # Zero-mean DC removal
        x_2ch = x_2ch - np.mean(x_2ch, axis=-1, keepdims=True)
        # Unit RMS normalization
        rms = np.sqrt(np.mean(x_2ch ** 2)) + eps
        x_2ch = x_2ch / rms
        # Length adjustment
        curr_len = x_2ch.shape[-1]
        if curr_len < target_length:
            pad = np.zeros((2, target_length - curr_len), dtype=np.float32)
            x_2ch = np.concatenate([x_2ch, pad], axis=-1)
        elif curr_len > target_length:
            x_2ch = x_2ch[:, :target_length]
        return torch.from_numpy(x_2ch.astype(np.float32))

    else:
        # Batch [B, 2, N]
        x_2ch = x_2ch - np.mean(x_2ch, axis=-1, keepdims=True)
        rms = np.sqrt(np.mean(x_2ch ** 2, axis=(-1, -2), keepdims=True)) + eps
        x_2ch = x_2ch / rms
        curr_len = x_2ch.shape[-1]
        if curr_len < target_length:
            pad = np.zeros((x_2ch.shape[0], 2, target_length - curr_len), dtype=np.float32)
            x_2ch = np.concatenate([x_2ch, pad], axis=-1)
        elif curr_len > target_length:
            x_2ch = x_2ch[:, :, :target_length]
        return torch.from_numpy(x_2ch.astype(np.float32))


# =====================================================================
# Standalone Fallback Architectures for Complete Portability
# =====================================================================

class _ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = (
            nn.Identity()
            if (stride == 1 and in_channels == out_channels)
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _ResNet1DBackbone(nn.Module):
    def __init__(self, num_classes: int = 8, input_channels: int = 2, base_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            _ResidualBlock1D(base_channels, base_channels, stride=1),
            _ResidualBlock1D(base_channels, base_channels * 2, stride=2),
            _ResidualBlock1D(base_channels * 2, base_channels * 4, stride=2),
            _ResidualBlock1D(base_channels * 4, base_channels * 4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        emb = self.pool(x).squeeze(-1)  # [B, 256]
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        emb_d = self.dropout(emb)
        logits = self.fc(emb_d)
        return logits


class _SpectrogramExtractor(nn.Module):
    def __init__(self, n_fft: int = 128, hop_length: int = 32, win_length: int = 128, include_phase: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.include_phase = include_phase
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, N]
        complex_iq = torch.complex(x[:, 0, :], x[:, 1, :])
        stft_out = torch.stft(
            complex_iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=False,
            return_complex=True,
        )
        stft_shifted = torch.fft.fftshift(stft_out, dim=1)
        power_spec = torch.abs(stft_shifted) ** 2
        log_power = 10.0 * torch.log10(power_spec + 1e-8)
        mean = log_power.mean(dim=(-2, -1), keepdim=True)
        std = log_power.std(dim=(-2, -1), keepdim=True) + 1e-6
        norm_power = (log_power - mean) / std

        if not self.include_phase:
            return norm_power.unsqueeze(1)

        phase = torch.angle(stft_shifted)
        d_phase = torch.diff(phase, dim=-1, prepend=phase[:, :, :1])
        d_phase_wrapped = (d_phase + math.pi) % (2.0 * math.pi) - math.pi
        d_phase_norm = d_phase_wrapped / math.pi
        return torch.stack([norm_power, d_phase_norm], dim=1)


class _SEAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, channels // reduction), channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.mean(dim=(-2, -1))
        w = self.fc2(self.relu(self.fc1(w)))
        w = self.sigmoid(w).view(b, c, 1, 1)
        return x * w


class _ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + res
        return self.relu(out)


class _SpectrogramCNNBackbone(nn.Module):
    def __init__(self, num_classes: int = 8, base_channels: int = 32, dropout: float = 0.25):
        super().__init__()
        self.num_classes = num_classes
        self.spectrogram_extractor = _SpectrogramExtractor(
            n_fft=128, hop_length=32, include_phase=False
        )
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        # Stage 1: 32 -> 64
        self.stage1 = nn.Sequential(
            _ResidualBlock2D(base_channels, 64, stride=2),
            _ResidualBlock2D(64, 64, stride=1),
        )
        # Stage 2: 64 -> 128
        self.stage2 = nn.Sequential(
            _ResidualBlock2D(64, 128, stride=2),
            _ResidualBlock2D(128, 128, stride=1),
        )
        # Stage 3: 128 -> 256
        self.stage3 = nn.Sequential(
            _ResidualBlock2D(128, 256, stride=2),
            _ResidualBlock2D(256, 256, stride=1),
        )
        self.attention = _SEAttention2D(256, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            spec = self.spectrogram_extractor(x)
        else:
            spec = x
        h = self.stem(spec)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.attention(h)
        feat = self.global_pool(h).flatten(1)  # [B, 256]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        logits = self.head(emb)
        return logits


# =====================================================================
# Branch Adapters
# =====================================================================

class BaseBranchAdapter(ABC):
    """Abstract interface for individual branch adapters."""

    @abstractmethod
    def predict_window(
        self,
        iq_window: Union[np.ndarray, torch.Tensor],
        source_signal_id: Optional[str] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
    ) -> BranchPrediction:
        pass

    @abstractmethod
    def predict_batch(
        self,
        iq_batch: Union[np.ndarray, torch.Tensor],
        source_signal_ids: Optional[List[str]] = None,
        window_starts: Optional[List[int]] = None,
        window_ends: Optional[List[int]] = None,
    ) -> List[BranchPrediction]:
        pass


class ResNet1DAdapter(BaseBranchAdapter):
    """
    Standardized adapter for the 1D ResNet raw-IQ modulation classifier.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
        model_name: str = "astra_resnet1d_v1",
        model_version: str = "1.0.0",
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.device = torch.device(device)
        self.scaler = TemperatureScaler(temperature=temperature)

        # Default standard V2 10-class alphabet
        self.class_names = class_names or list(TRAINED_MODULATION_CLASSES_V2)

        if model is not None:
            self.model = model.to(self.device)
        else:
            try:
                from ASTRA_Modulation_Model_Starter.ASTRA_Modulation_Model_Starter.src.model import ResNet1DModClassifier
                self.model = ResNet1DModClassifier(num_classes=len(self.class_names)).to(self.device)
            except Exception:
                self.model = _ResNet1DBackbone(num_classes=len(self.class_names)).to(self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)

        self.model.eval()

    def _load_checkpoint(self, path: str) -> None:
        from astra_config.classes import MODULATION_ALIASES
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict):
            raw_c = ckpt.get("classes", ckpt.get("class_names"))
            if raw_c:
                self.class_names = [MODULATION_ALIASES.get(str(c).lower().strip(), str(c).upper()) for c in raw_c]
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            self.model.load_state_dict(state_dict, strict=False)
        elif isinstance(ckpt, nn.Module):
            self.model = ckpt.to(self.device)

    def predict_window(
        self,
        iq_window: Union[np.ndarray, torch.Tensor],
        source_signal_id: Optional[str] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
    ) -> BranchPrediction:
        x_tensor = preprocess_iq(iq_window, target_length=2048)  # [2, 2048]
        x_batch = x_tensor.unsqueeze(0).to(self.device)          # [1, 2, 2048]

        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                emb = self.model.forward_features(x_batch)
                logits = self.model.fc(self.model.dropout(emb)) if hasattr(self.model, "fc") else self.model(x_batch)
            else:
                logits = self.model(x_batch)
                emb = torch.zeros((1, 256), device=self.device)

            scaled_logits = self.scaler.scale_logits(logits)
            probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()
            logits_list = logits.squeeze(0).cpu().tolist()
            emb_list = emb.squeeze(0).cpu().tolist()

        sorted_indices = np.argsort(probs)[::-1]
        best_idx = int(sorted_indices[0])
        predicted_class = self.class_names[best_idx]
        confidence = float(probs[best_idx])

        top_k = [
            {"rank": r + 1, "class": self.class_names[idx], "probability": float(probs[idx])}
            for r, idx in enumerate(sorted_indices[:3])
        ]

        return BranchPrediction(
            model_name=self.model_name,
            model_version=self.model_version,
            class_names=self.class_names,
            logits=logits_list,
            probabilities=probs.tolist(),
            predicted_class=predicted_class,
            confidence=confidence,
            top_k=top_k,
            feature_embedding=emb_list,
            source_signal_id=source_signal_id,
            window_start=window_start,
            window_end=window_end,
        )

    def predict_batch(
        self,
        iq_batch: Union[np.ndarray, torch.Tensor],
        source_signal_ids: Optional[List[str]] = None,
        window_starts: Optional[List[int]] = None,
        window_ends: Optional[List[int]] = None,
    ) -> List[BranchPrediction]:
        x_tensor = preprocess_iq(iq_batch, target_length=2048).to(self.device)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        batch_size = x_tensor.size(0)
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                emb = self.model.forward_features(x_tensor)
                logits = self.model.fc(self.model.dropout(emb)) if hasattr(self.model, "fc") else self.model(x_tensor)
            else:
                logits = self.model(x_tensor)
                emb = torch.zeros((batch_size, 256), device=self.device)

            scaled_logits = self.scaler.scale_logits(logits)
            probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()
            logits_arr = logits.cpu().numpy()
            emb_arr = emb.cpu().numpy()

        results: List[BranchPrediction] = []
        for i in range(batch_size):
            p = probs[i]
            sorted_indices = np.argsort(p)[::-1]
            best_idx = int(sorted_indices[0])
            top_k = [
                {"rank": r + 1, "class": self.class_names[idx], "probability": float(p[idx])}
                for r, idx in enumerate(sorted_indices[:3])
            ]
            results.append(
                BranchPrediction(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    class_names=self.class_names,
                    logits=logits_arr[i].tolist(),
                    probabilities=p.tolist(),
                    predicted_class=self.class_names[best_idx],
                    confidence=float(p[best_idx]),
                    top_k=top_k,
                    feature_embedding=emb_arr[i].tolist(),
                    source_signal_id=source_signal_ids[i] if source_signal_ids else None,
                    window_start=window_starts[i] if window_starts else None,
                    window_end=window_ends[i] if window_ends else None,
                )
            )
        return results


class Spectrogram2DAdapter(BaseBranchAdapter):
    """
    Standardized adapter for the 2D Spectrogram CNN modulation classifier.
    Computes baseband STFT spectrogram directly from the input IQ window.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
        model_name: str = "astra_spectrogram2d_v1",
        model_version: str = "1.0.0",
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.device = torch.device(device)
        self.scaler = TemperatureScaler(temperature=temperature)

        # Default canonical standard V2 10-class alphabet
        self.class_names = class_names or list(TRAINED_MODULATION_CLASSES_V2)
        self.internal_classes = list(self.class_names)

        if model is not None:
            self.model = model.to(self.device)
        else:
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)
            else:
                try:
                    from astra_modulation_2d.src.model import ASTRASpectrogramCNN
                    self.model = ASTRASpectrogramCNN(class_names=self.internal_classes).to(self.device)
                except Exception:
                    self.model = _SpectrogramCNNBackbone(num_classes=len(self.internal_classes)).to(self.device)

        self.model.eval()

    def _load_checkpoint(self, path: str) -> None:
        from astra_config.classes import MODULATION_ALIASES
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict):
            raw_classes = ckpt.get("class_names", ckpt.get("classes", []))
            if raw_classes:
                self.internal_classes = [MODULATION_ALIASES.get(str(c).lower(), str(c).upper()) for c in raw_classes]
            try:
                from astra_modulation_2d.src.model import ASTRASpectrogramCNN
                self.model = ASTRASpectrogramCNN(class_names=raw_classes if raw_classes else self.internal_classes).to(self.device)
            except Exception:
                self.model = _SpectrogramCNNBackbone(num_classes=len(self.internal_classes)).to(self.device)

            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            self.model.load_state_dict(state_dict, strict=False)
        elif isinstance(ckpt, nn.Module):
            self.model = ckpt.to(self.device)

    def predict_window(
        self,
        iq_window: Union[np.ndarray, torch.Tensor],
        source_signal_id: Optional[str] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
    ) -> BranchPrediction:
        x_tensor = preprocess_iq(iq_window, target_length=2048)  # [2, 2048]
        x_batch = x_tensor.unsqueeze(0).to(self.device)          # [1, 2, 2048]

        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "extract_features"):
                emb = self.model.extract_features(x_batch)
                logits = self.model(x_batch)
            elif hasattr(self.model, "forward_features"):
                emb = self.model.forward_features(x_batch)
                logits = self.model(x_batch)
            else:
                logits = self.model(x_batch)
                emb = torch.zeros((1, 256), device=self.device)

            scaled_logits = self.scaler.scale_logits(logits)
            raw_probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()
            emb_list = emb.squeeze(0).cpu().tolist()

        # Map internal class probabilities into canonical 10-class vector
        probs_10 = np.zeros(len(self.class_names), dtype=np.float32)
        logits_10 = [-10.0] * len(self.class_names)
        raw_logits = logits.squeeze(0).cpu().numpy()
        for i_int, c_name in enumerate(self.internal_classes):
            if c_name in self.class_names:
                idx_10 = self.class_names.index(c_name)
                probs_10[idx_10] = raw_probs[i_int]
                logits_10[idx_10] = float(raw_logits[i_int])

        # Renormalize probabilities if any valid classes matched
        s = float(np.sum(probs_10))
        if s > 1e-6:
            probs_10 = probs_10 / s

        sorted_indices = np.argsort(probs_10)[::-1]
        best_idx = int(sorted_indices[0])
        predicted_class = self.class_names[best_idx]
        confidence = float(probs_10[best_idx])

        top_k = [
            {"rank": r + 1, "class": self.class_names[idx], "probability": float(probs_10[idx])}
            for r, idx in enumerate(sorted_indices[:3])
        ]

        return BranchPrediction(
            model_name=self.model_name,
            model_version=self.model_version,
            class_names=self.class_names,
            logits=logits_10,
            probabilities=probs_10.tolist(),
            predicted_class=predicted_class,
            confidence=confidence,
            top_k=top_k,
            feature_embedding=emb_list,
            source_signal_id=source_signal_id,
            window_start=window_start,
            window_end=window_end,
        )

    def predict_batch(
        self,
        iq_batch: Union[np.ndarray, torch.Tensor],
        source_signal_ids: Optional[List[str]] = None,
        window_starts: Optional[List[int]] = None,
        window_ends: Optional[List[int]] = None,
    ) -> List[BranchPrediction]:
        x_tensor = preprocess_iq(iq_batch, target_length=2048).to(self.device)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        batch_size = x_tensor.size(0)
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "extract_features"):
                emb = self.model.extract_features(x_tensor)
                logits = self.model(x_tensor)
            elif hasattr(self.model, "forward_features"):
                emb = self.model.forward_features(x_tensor)
                logits = self.model(x_tensor)
            else:
                logits = self.model(x_tensor)
                emb = torch.zeros((batch_size, 256), device=self.device)

            scaled_logits = self.scaler.scale_logits(logits)
            raw_probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()
            raw_logits = logits.cpu().numpy()
            emb_arr = emb.cpu().numpy()

        results: List[BranchPrediction] = []
        for i in range(batch_size):
            probs_10 = np.zeros(len(self.class_names), dtype=np.float32)
            logits_10 = [-10.0] * len(self.class_names)
            for i_int, c_name in enumerate(self.internal_classes):
                if c_name in self.class_names:
                    idx_10 = self.class_names.index(c_name)
                    probs_10[idx_10] = raw_probs[i, i_int]
                    logits_10[idx_10] = float(raw_logits[i, i_int])

            s = float(np.sum(probs_10))
            if s > 1e-6:
                probs_10 = probs_10 / s

            sorted_indices = np.argsort(probs_10)[::-1]
            best_idx = int(sorted_indices[0])
            top_k = [
                {"rank": r + 1, "class": self.class_names[idx], "probability": float(probs_10[idx])}
                for r, idx in enumerate(sorted_indices[:3])
            ]
            results.append(
                BranchPrediction(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    class_names=self.class_names,
                    logits=logits_10,
                    probabilities=probs_10.tolist(),
                    predicted_class=self.class_names[best_idx],
                    confidence=float(probs_10[best_idx]),
                    top_k=top_k,
                    feature_embedding=emb_arr[i].tolist(),
                    source_signal_id=source_signal_ids[i] if source_signal_ids else None,
                    window_start=window_starts[i] if window_starts else None,
                    window_end=window_ends[i] if window_ends else None,
                )
            )
        return results
