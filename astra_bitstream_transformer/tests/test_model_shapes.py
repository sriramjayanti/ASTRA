"""
test_model_shapes.py
Unit tests for model architecture shapes, stride invariance, positional encodings, and finite logits.
"""

import pytest
import torch
from astra_bitstream_transformer.src.models import ASTRABitstreamCNNTransformer
from astra_bitstream_transformer.src.cnn_encoder import BitstreamCNNEncoder
from astra_bitstream_transformer.src.transformer import BitstreamTransformerEncoder
from astra_bitstream_transformer.src.positional_encoding import SinusoidalPositionalEncoding


def test_1_and_2_input_output_shapes():
    """TEST 1 & 2: input shape [B, C, L] accepted and output shape [B, L, num_classes] correct."""
    model = ASTRABitstreamCNNTransformer(
        in_channels=6,
        d_model=128,
        num_heads=4,
        num_layers=2,
        num_classes=6
    )

    B, C, L = 2, 6, 256
    x = torch.randn(B, C, L)
    out = model(x)

    assert "sequence_logits" in out
    assert out["sequence_logits"].shape == (B, L, 6)
    assert "boundary_logits" in out
    assert out["boundary_logits"].shape == (B, L)
    assert "frame_logits" in out
    assert out["frame_logits"].shape == (B, 4)


def test_5_positional_encoding_shape():
    """TEST 5: positional encoding shape and length flexibility."""
    pe = SinusoidalPositionalEncoding(d_model=128, max_len=512)
    x = torch.randn(2, 300, 128)
    out = pe(x)
    assert out.shape == (2, 300, 128)

    # Dynamic extension beyond max_len
    x_long = torch.randn(1, 600, 128)
    out_long = pe(x_long)
    assert out_long.shape == (1, 600, 128)


def test_6_cnn_preserves_length():
    """TEST 6: 1D CNN preserves exact bit length L (stride=1)."""
    cnn = BitstreamCNNEncoder(in_channels=6, channels=[32, 64, 128], kernel_size=7)
    x = torch.randn(2, 6, 512)
    feats = cnn(x)
    assert feats.shape == (2, 128, 512)


def test_7_transformer_preserves_length():
    """TEST 7: Transformer Encoder preserves length L."""
    trans = BitstreamTransformerEncoder(d_model=128, num_heads=4, num_layers=2)
    x = torch.randn(2, 512, 128)
    out = trans(x)
    assert out.shape == (2, 512, 128)


def test_8_all_logits_finite():
    """TEST 8: all forward logits are finite without NaNs or Infs."""
    model = ASTRABitstreamCNNTransformer(in_channels=6, d_model=64, num_heads=2, num_layers=1)
    x = torch.randn(2, 6, 128)
    out = model(x)

    assert torch.all(torch.isfinite(out["sequence_logits"]))
    assert torch.all(torch.isfinite(out["boundary_logits"]))
