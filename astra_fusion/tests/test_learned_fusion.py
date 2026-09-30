"""
Test suite for Mode B Learned Fusion and checkpoint serialization.
"""

import os
import tempfile
import torch
import pytest
from astra_fusion.src.checkpoint import load_fusion_checkpoint, save_fusion_checkpoint
from astra_fusion.src.learned_fusion import LearnedFusionEngine, LearnedFusionMLP
from astra_fusion.src.models import BranchPrediction


def test_19_learned_fusion_and_checkpoint_reload():
    """TEST 19: Learned MLP forward pass, checkpoint saving, and exact state reload."""
    classes = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
    mlp = LearnedFusionMLP(dim_1d_feature=256, dim_2d_feature=256, num_classes=8)

    # 1. Test forward pass
    f1 = torch.randn(4, 256)
    f2 = torch.randn(4, 256)
    l1 = torch.randn(4, 8)
    l2 = torch.randn(4, 8)

    logits, emb = mlp(f1, f2, l1, l2)
    assert logits.shape == (4, 8)
    assert emb.shape == (4, 128)

    # 2. Test Checkpoint save & reload
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "fusion_ckpt.pt")
        save_fusion_checkpoint(
            path=ckpt_path,
            fusion_model=mlp,
            class_names=classes,
            branch_versions={"1d": "1.0", "2d": "1.0"},
            validation_metrics={"val_acc": 0.95},
        )
        assert os.path.exists(ckpt_path)

        loaded_model, meta = load_fusion_checkpoint(ckpt_path)
        assert meta["validation_metrics"]["val_acc"] == 0.95
        assert meta["class_names"] == classes

        # Compare outputs
        mlp.eval()
        loaded_model.eval()
        with torch.no_grad():
            out_orig, _ = mlp(f1, f2, l1, l2)
            out_loaded, _ = loaded_model(f1, f2, l1, l2)
            assert torch.allclose(out_orig, out_loaded, atol=1e-5)
