"""
train.py
Training pipeline with AdamW, learning rate scheduling, gradient clipping, tiny-overfit verification, and early stopping.
"""

from typing import Dict, Any, Optional
import os
import torch
from torch.utils.data import DataLoader

from .models import ASTRABitstreamCNNTransformer
from .losses import CombinedBitstreamLoss
from .evaluate import evaluate_model
from .utils import save_checkpoint


def run_tiny_overfit_test(
    model: torch.nn.Module,
    tiny_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    num_steps: int = 60,
    lr: float = 0.001
) -> bool:
    """
    Verify model capacity and gradient flow by overfitting a tiny batch of 2-4 sequences.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = CombinedBitstreamLoss()

    batch = next(iter(tiny_loader))
    x = batch["channels"].to(device)
    mask = batch["padding_mask"].to(device)
    labels = batch["labels"].to(device)
    b_targets = batch["boundary_targets"].to(device)

    initial_loss = float("inf")
    final_loss = float("inf")

    for step in range(num_steps):
        optimizer.zero_grad()
        out = model(x, padding_mask=mask)
        losses = loss_fn(
            out,
            targets=labels,
            padding_mask=mask,
            boundary_targets=b_targets
        )
        loss = losses["total_loss"]
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    # Success if loss dropped substantially
    success = (final_loss < 0.20) or (final_loss < 0.3 * initial_loss)
    return success


def train_bitstream_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    checkpoint_dir: str = "checkpoints/bitstream_transformer",
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Train ASTRABitstreamCNNTransformer model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    lr = config.get("learning_rate", 0.0003)
    weight_decay = config.get("weight_decay", 0.0001)
    epochs = config.get("epochs", 20)
    grad_clip = config.get("grad_clip_norm", 1.0)
    patience = config.get("early_stopping_patience", 7)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Class weights for loss
    class_weights = None
    if hasattr(train_loader.dataset, "compute_class_weights"):
        class_weights = train_loader.dataset.compute_class_weights().to(device)

    b_weight = config.get("loss", {}).get("boundary_loss_weight", 0.15) if isinstance(config.get("loss"), dict) else 0.15
    loss_fn = CombinedBitstreamLoss(
        class_weights=class_weights,
        boundary_loss_weight=b_weight
    )

    best_macro_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["channels"].to(device)
            mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)
            b_targets = batch["boundary_targets"].to(device)
            f_targets = batch.get("frame_targets")
            if f_targets is not None:
                f_targets = f_targets.to(device)

            optimizer.zero_grad()
            out = model(x, padding_mask=mask)
            losses = loss_fn(
                out,
                targets=labels,
                padding_mask=mask,
                boundary_targets=b_targets,
                frame_targets=f_targets
            )

            loss = losses["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_train_loss = running_loss / max(1, n_batches)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device=device)
        macro_f1 = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_macro_f1": macro_f1,
            "val_accuracy": val_metrics["accuracy"]
        })

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            save_checkpoint(
                model=model,
                checkpoint_path=best_checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                config=config
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return {
        "best_macro_f1": best_macro_f1,
        "best_checkpoint_path": best_checkpoint_path,
        "history": history
    }
