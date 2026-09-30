import tempfile
from pathlib import Path
import torch

from astra_modulation_2d.src.model import ASTRASpectrogramCNN
from astra_modulation_2d.src.checkpoint import save_checkpoint, load_checkpoint


def test_14_checkpoint_save_and_load():
    """TEST 14: Checkpoint saves and restores model weights accurately."""
    classes = ["bpsk", "qpsk", "8psk"]
    model1 = ASTRASpectrogramCNN(class_names=classes)
    opt = torch.optim.Adam(model1.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_p = Path(tmp_dir) / "test_ckpt.pt"
        save_checkpoint(
            filepath=ckpt_p,
            model=model1,
            optimizer=opt,
            epoch=5,
            best_metric=0.85,
            class_names=classes,
        )

        model2 = ASTRASpectrogramCNN(class_names=classes)
        meta = load_checkpoint(ckpt_p, model2)

        assert meta["epoch"] == 5
        assert meta["best_metric"] == 0.85
        # Verify state dict matches
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_15_class_mapping_preserved_and_mismatch_fails():
    """TEST 15: Incompatible class ordering or mismatch throws an error on checkpoint load."""
    classes1 = ["bpsk", "qpsk", "8psk"]
    model1 = ASTRASpectrogramCNN(class_names=classes1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_p = Path(tmp_dir) / "test_ckpt.pt"
        save_checkpoint(ckpt_p, model1, class_names=classes1)

        # Incompatible model with 4 classes
        classes2 = ["bpsk", "qpsk", "8psk", "16qam"]
        model2 = ASTRASpectrogramCNN(class_names=classes2)

        failed = False
        try:
            load_checkpoint(ckpt_p, model2, strict_class_check=True)
        except ValueError:
            failed = True
        assert failed, "Expected ValueError on class count mismatch."
