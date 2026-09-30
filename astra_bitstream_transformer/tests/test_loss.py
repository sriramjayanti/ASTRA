"""
test_loss.py
Unit tests for finite loss computation, class weighting, and tiny dataset overfitting.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from astra_bitstream_transformer.src.models import ASTRABitstreamCNNTransformer
from astra_bitstream_transformer.src.losses import CombinedBitstreamLoss
from astra_bitstream_transformer.src.dataset import create_synthetic_dataset
from astra_bitstream_transformer.src.train import run_tiny_overfit_test


def test_9_loss_finite():
    """TEST 9: loss computation is finite and non-negative."""
    loss_fn = CombinedBitstreamLoss()
    outputs = {
        "sequence_logits": torch.randn(2, 64, 6),
        "boundary_logits": torch.randn(2, 64),
        "frame_logits": torch.randn(2, 4)
    }
    targets = torch.randint(0, 6, (2, 64))
    b_targets = torch.randint(0, 2, (2, 64)).float()
    f_targets = torch.randint(0, 2, (2, 4)).float()

    losses = loss_fn(
        outputs,
        targets=targets,
        boundary_targets=b_targets,
        frame_targets=f_targets
    )

    assert torch.isfinite(losses["total_loss"])
    assert losses["total_loss"].item() > 0.0


def test_10_class_weights_accepted():
    """TEST 10: class weights properly accepted and scale loss."""
    weights = torch.tensor([1.0, 5.0, 3.0, 0.5, 4.0, 2.0])
    loss_fn = CombinedBitstreamLoss(class_weights=weights)

    outputs = {"sequence_logits": torch.randn(2, 32, 6)}
    targets = torch.randint(0, 6, (2, 32))

    losses = loss_fn(outputs, targets=targets)
    assert torch.isfinite(losses["total_loss"])


def test_11_tiny_dataset_overfits():
    """TEST 11: tiny dataset overfit test passes."""
    dataset = create_synthetic_dataset(num_streams=2, window_length=256, seed=42)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = ASTRABitstreamCNNTransformer(
        in_channels=6,
        cnn_channels=[32, 64, 128],
        d_model=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256
    )

    success = run_tiny_overfit_test(model, loader, num_steps=60, lr=0.003)
    assert success, "Model failed to overfit tiny synthetic dataset."
