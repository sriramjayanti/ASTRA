"""
positional_encoding.py
Sinusoidal positional encoding module for variable-length bitstream sequences.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence modeling.
    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int = 256, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model] with positional encoding added.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Dynamic extension if sequence exceeds precomputed max_len
            d_model = x.size(2)
            pe_ext = torch.zeros(seq_len, d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * (-math.log(10000.0) / d_model))
            pe_ext[:, 0::2] = torch.sin(position * div_term)
            pe_ext[:, 1::2] = torch.cos(position * div_term)
            pos_emb = pe_ext.unsqueeze(0)
        else:
            pos_emb = self.pe[:, :seq_len]

        x = x + pos_emb
        return self.dropout(x)
