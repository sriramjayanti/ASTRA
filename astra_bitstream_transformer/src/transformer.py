"""
transformer.py
Transformer Encoder stack with padding mask support for long-range sequence context modeling.
"""

from typing import Optional
import torch
import torch.nn as nn


class BitstreamTransformerEncoder(nn.Module):
    """
    Transformer Encoder stack for capturing long-range contextual bitstream relationships.
    Input shape:  [B, L, d_model]
    Output shape: [B, L, d_model]
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence tensor [B, L, d_model]
            src_key_padding_mask: Optional boolean tensor [B, L] where True indicates padded positions.

        Returns:
            Contextualized sequence tensor [B, L, d_model]
        """
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out
