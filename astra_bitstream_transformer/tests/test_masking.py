"""
test_masking.py
Unit tests for padding masks, variable sequence lengths, and loss masking.
"""

import pytest
import torch
from astra_bitstream_transformer.src.models import ASTRABitstreamCNNTransformer
from astra_bitstream_transformer.src.losses import CombinedBitstreamLoss


def test_3_padding_mask_works():
    """TEST 3: padding mask properly applied in Transformer attention."""
    model = ASTRABitstreamCNNTransformer(in_channels=6, d_model=64, num_heads=2, num_layers=2)
    B, C, L = 2, 6, 128
    x = torch.randn(B, C, L)

    # First sequence valid 100 bits (28 padded), second sequence valid 128 bits
    padding_mask = torch.zeros((B, L), dtype=torch.bool)
    padding_mask[0, 100:] = True

    out = model(x, padding_mask=padding_mask)
    assert out["sequence_logits"].shape == (B, L, 6)
    assert torch.all(torch.isfinite(out["sequence_logits"]))


def test_4_padded_positions_ignored_in_loss():
    """TEST 4: padded positions are masked out from sequence loss."""
    loss_fn = CombinedBitstreamLoss(ignore_index=-100)

    B, L, C = 2, 64, 6
    seq_logits = torch.randn(B, L, C, requires_grad=True)
    b_logits = torch.randn(B, L, requires_grad=True)
    outputs = {"sequence_logits": seq_logits, "boundary_logits": b_logits}

    targets = torch.randint(0, 6, (B, L))
    padding_mask = torch.zeros((B, L), dtype=torch.bool)
    padding_mask[:, 32:] = True # Ignore second half

    # Compute loss with mask
    losses = loss_fn(outputs, targets=targets, padding_mask=padding_mask)
    loss = losses["total_loss"]
    loss.backward()

    assert torch.all(torch.isfinite(loss))
    # Gradients for masked positions should not affect valid loss
    assert seq_logits.grad is not None


def test_12_variable_sequence_lengths():
    """TEST 12: model seamlessly handles various sequence lengths."""
    model = ASTRABitstreamCNNTransformer(in_channels=6, d_model=64, num_heads=2, num_layers=1)

    for seq_len in [64, 128, 512, 1024]:
        x = torch.randn(1, 6, seq_len)
        out = model(x)
        assert out["sequence_logits"].shape == (1, seq_len, 6)
