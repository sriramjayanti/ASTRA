"""
models.py
Data models, enums, label definitions, prediction dataclasses, and the core ASTRABitstreamCNNTransformer architecture.
"""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from .cnn_encoder import BitstreamCNNEncoder
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer import BitstreamTransformerEncoder
from .heads import SequenceClassificationHead, BoundaryPredictionHead, FrameClassificationHead


class StructureLabel(IntEnum):
    UNKNOWN = 0
    SYNC = 1
    HEADER = 2
    PAYLOAD = 3
    CRC = 4
    PADDING = 5


LABEL_NAMES: List[str] = [
    "UNKNOWN",
    "SYNC",
    "HEADER",
    "PAYLOAD",
    "CRC",
    "PADDING"
]

LABEL_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL: Dict[int, str] = {i: name for i, name in enumerate(LABEL_NAMES)}


@dataclass
class RegionPrediction:
    """Contiguous identified structural region in the bitstream."""
    start_bit: int
    end_bit: int
    label: str
    label_id: int
    mean_probability: float
    min_probability: float
    bit_length: int = field(init=False)

    def __post_init__(self):
        self.bit_length = self.end_bit - self.start_bit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_bit": int(self.start_bit),
            "end_bit": int(self.end_bit),
            "bit_length": int(self.bit_length),
            "label": str(self.label),
            "label_id": int(self.label_id),
            "mean_probability": round(float(self.mean_probability), 4),
            "min_probability": round(float(self.min_probability), 4),
        }


@dataclass
class BitstreamStructurePrediction:
    """
    Comprehensive structure prediction outcome for a recovered bitstream from Stage 13.
    Feeds predicted regions and boundary markers to Stage 14 Header / Payload Explorer.
    """
    pipeline_path_id: str
    sequence_length: int

    # Bit-level predictions
    predicted_labels: np.ndarray # Shape [L] (int array of label IDs)
    label_probabilities: np.ndarray # Shape [L, num_classes] (float32)
    label_logits: Optional[np.ndarray] = None # Shape [L, num_classes]

    # Contiguous parsed regions & transition boundaries
    regions: List[RegionPrediction] = field(default_factory=list)
    boundary_positions: List[int] = field(default_factory=list)

    # Frame-level multi-label predictions
    frame_level_predictions: Dict[str, float] = field(default_factory=dict)

    # Uncertainty & Overall Metrics
    model_confidence: float = 0.0
    mean_uncertainty: float = 0.0
    uncertain_regions: List[Dict[str, Any]] = field(default_factory=list)

    # Lineage and Metadata
    model_version: str = "astra_bitstream_cnn_transformer_v1"
    feature_schema_version: str = "bitstream_model_input_v1"

    def to_dict(self, include_per_bit: bool = False) -> Dict[str, Any]:
        data = {
            "pipeline_path_id": str(self.pipeline_path_id),
            "sequence_length": int(self.sequence_length),
            "model_version": str(self.model_version),
            "feature_schema_version": str(self.feature_schema_version),
            "model_confidence": round(float(self.model_confidence), 4),
            "mean_uncertainty": round(float(self.mean_uncertainty), 4),
            "regions": [r.to_dict() for r in self.regions],
            "boundary_positions": [int(b) for b in self.boundary_positions[:20]],
            "frame_level_predictions": {k: round(float(v), 4) for k, v in self.frame_level_predictions.items()},
            "uncertain_regions": self.uncertain_regions[:10],
        }
        if include_per_bit:
            data["predicted_labels"] = self.predicted_labels.tolist()
            data["label_probabilities"] = self.label_probabilities.tolist()
        return data


class ASTRABitstreamCNNTransformer(nn.Module):
    """
    ASTRA Stage 13: 1D CNN + Transformer Neural Sequence Model for Bitstream Structure Classification.

    Architecture Flow:
      Input [B, C, L]
        ↓
      1D CNN Local Feature Extractor (Stride=1, preserves bit resolution) -> [B, D, L]
        ↓ Transpose -> [B, L, D]
      Sinusoidal Positional Encoding -> [B, L, D]
        ↓
      Transformer Encoder Stack (Self-Attention + Padding Mask) -> [B, L, D]
        ↓
      Per-position Sequence Head -> [B, L, num_classes]
      Auxiliary Boundary Head    -> [B, L]
      Masked Frame-level Head    -> [B, num_frame_classes]
    """

    def __init__(
        self,
        in_channels: int = 6,
        cnn_channels: Optional[List[int]] = None,
        cnn_kernel_size: int = 7,
        cnn_residual_blocks: int = 2,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        num_classes: int = 6,
        enable_frame_head: bool = True,
        num_frame_classes: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 8192
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, d_model]

        self.in_channels = in_channels
        self.d_model = d_model
        self.num_classes = num_classes
        self.enable_frame_head = enable_frame_head

        # 1. 1D CNN Local Feature Extractor
        self.cnn_encoder = BitstreamCNNEncoder(
            in_channels=in_channels,
            channels=cnn_channels,
            kernel_size=cnn_kernel_size,
            residual_blocks_per_stage=cnn_residual_blocks,
            dropout=dropout
        )

        # 2. Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )

        # 3. Transformer Encoder
        self.transformer_encoder = BitstreamTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # 4. Heads
        self.sequence_head = SequenceClassificationHead(
            d_model=d_model,
            hidden_dim=d_model // 2,
            num_classes=num_classes,
            dropout=dropout
        )

        self.boundary_head = BoundaryPredictionHead(
            d_model=d_model,
            hidden_dim=64,
            dropout=dropout
        )

        if enable_frame_head:
            self.frame_head = FrameClassificationHead(
                d_model=d_model,
                hidden_dim=d_model // 2,
                num_frame_classes=num_frame_classes,
                dropout=dropout
            )
        else:
            self.frame_head = None

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for bitstream structure prediction.

        Args:
            x: Input tensor of shape [B, C, L]
            padding_mask: Boolean tensor of shape [B, L] where True = padded bit

        Returns:
            Dict containing:
              - 'sequence_logits': [B, L, num_classes]
              - 'boundary_logits': [B, L]
              - 'frame_logits': [B, num_frame_classes] (if enabled)
        """
        # 1. Local CNN motifs [B, C, L] -> [B, D, L]
        cnn_feats = self.cnn_encoder(x)

        # 2. Transpose to [B, L, D] and add positional encoding
        seq_feats = cnn_feats.transpose(1, 2)
        seq_feats = self.pos_encoder(seq_feats)

        # 3. Transformer contextual attention [B, L, D]
        context_feats = self.transformer_encoder(seq_feats, src_key_padding_mask=padding_mask)

        # 4. Predictions
        seq_logits = self.sequence_head(context_feats)
        boundary_logits = self.boundary_head(context_feats)

        out = {
            "sequence_logits": seq_logits,
            "boundary_logits": boundary_logits,
            "embeddings": context_feats
        }

        if self.frame_head is not None:
            frame_logits = self.frame_head(context_feats, padding_mask=padding_mask)
            out["frame_logits"] = frame_logits

        return out

    def extract_features(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract latent representations:
        Returns:
            (sequence_embeddings [B, L, D], frame_embedding [B, D])
        """
        cnn_feats = self.cnn_encoder(x)
        seq_feats = cnn_feats.transpose(1, 2)
        seq_feats = self.pos_encoder(seq_feats)
        context_feats = self.transformer_encoder(seq_feats, src_key_padding_mask=padding_mask)

        frame_emb = None
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            sum_x = torch.sum(context_feats * valid_mask, dim=1)
            counts = torch.clamp(valid_mask.sum(dim=1), min=1.0)
            frame_emb = sum_x / counts
        else:
            frame_emb = torch.mean(context_feats, dim=1)

        return context_feats, frame_emb
