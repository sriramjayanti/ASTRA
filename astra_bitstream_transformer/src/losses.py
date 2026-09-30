"""
losses.py
Loss functions for sequence labeling, class imbalance handling, and boundary/frame auxiliary supervision.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedBitstreamLoss(nn.Module):
    """
    Combined loss for Stage 13 sequence structure modeling:
      Total = L_sequence (Weighted CrossEntropy)
            + lambda_boundary * L_boundary (BCEWithLogits)
            + lambda_frame * L_frame (BCEWithLogits / CrossEntropy)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        boundary_loss_weight: float = 0.15,
        frame_loss_weight: float = 0.10
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.boundary_loss_weight = boundary_loss_weight
        self.frame_loss_weight = frame_loss_weight

        self.seq_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        self.boundary_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.frame_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        boundary_targets: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs dict containing 'sequence_logits', 'boundary_logits', 'frame_logits'
            targets: Target label sequence [B, L]
            padding_mask: Boolean tensor [B, L] (True = padded)
            boundary_targets: Binary tensor [B, L] (1.0 at boundary transition)
            frame_targets: Binary tensor [B, num_frame_classes]

        Returns:
            Dict containing 'total_loss', 'sequence_loss', 'boundary_loss', 'frame_loss'
        """
        seq_logits = outputs["sequence_logits"] # [B, L, C]
        B, L, C = seq_logits.shape

        # 1. Sequence Cross-Entropy Loss
        # Mask out padded positions if not already set to ignore_index
        clean_targets = targets.clone()
        if padding_mask is not None:
            clean_targets[padding_mask] = self.ignore_index

        seq_loss = self.seq_loss_fn(seq_logits.view(-1, C), clean_targets.view(-1))

        # 2. Auxiliary Boundary Loss
        boundary_loss = torch.tensor(0.0, device=seq_logits.device)
        if boundary_targets is not None and "boundary_logits" in outputs:
            b_logits = outputs["boundary_logits"] # [B, L]
            b_loss_raw = self.boundary_loss_fn(b_logits, boundary_targets.float())
            if padding_mask is not None:
                valid_mask = (~padding_mask).float()
                boundary_loss = (b_loss_raw * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)
            else:
                boundary_loss = b_loss_raw.mean()

        # 3. Auxiliary Frame Loss
        frame_loss = torch.tensor(0.0, device=seq_logits.device)
        if frame_targets is not None and "frame_logits" in outputs:
            frame_logits = outputs["frame_logits"]
            frame_loss = self.frame_loss_fn(frame_logits, frame_targets.float())

        total_loss = (
            seq_loss
            + self.boundary_loss_weight * boundary_loss
            + self.frame_loss_weight * frame_loss
        )

        return {
            "total_loss": total_loss,
            "sequence_loss": seq_loss,
            "boundary_loss": boundary_loss,
            "frame_loss": frame_loss
        }
