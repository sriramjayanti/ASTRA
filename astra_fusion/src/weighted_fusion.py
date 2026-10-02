"""
ASTRA Mode A: Weighted Probability & Logit Fusion Module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from .confidence import ConfidenceCalculator
from .statistical_gating import apply_statistical_prior_gating
from .models import (
    BranchPrediction,
    CandidateItem,
    FusionPrediction,
    InvalidWeightError,
)
from .validation import (
    validate_class_alignment,
    validate_probabilities,
    validate_source_alignment,
    validate_weights,
)


class WeightedProbabilityFusion:
    """
    Combines prediction probabilities from the 1D ResNet and 2D Spectrogram CNN branches
    using configurable convex weights:
        P_fused = w_1d * P_1d + w_2d * P_2d
    """

    def __init__(
        self,
        weight_1d: float = 0.70,
        weight_2d: float = 0.30,
        top_k: int = 3,
        confidence_calc: Optional[ConfidenceCalculator] = None,
    ):
        validate_weights(weight_1d, weight_2d)
        self.weight_1d = float(weight_1d)
        self.weight_2d = float(weight_2d)
        self.top_k = int(top_k)
        self.confidence_calc = confidence_calc or ConfidenceCalculator()

    def set_weights(self, weight_1d: float, weight_2d: float) -> None:
        """Dynamically updates the branch fusion weights."""
        validate_weights(weight_1d, weight_2d)
        self.weight_1d = float(weight_1d)
        self.weight_2d = float(weight_2d)

    def fuse_probabilities(
        self,
        probs_1d: np.ndarray,
        probs_2d: np.ndarray,
    ) -> np.ndarray:
        """
        Fuses 1D and 2D probability vectors.
        Shapes: [..., num_classes]
        """
        p1 = validate_probabilities(probs_1d)
        p2 = validate_probabilities(probs_2d)
        if p1.shape != p2.shape:
            raise ValueError(f"Shape mismatch in probabilities: 1D {p1.shape} vs 2D {p2.shape}")

        fused = self.weight_1d * p1 + self.weight_2d * p2
        # Normalize slightly to counter float precision drift
        fused = fused / np.sum(fused, axis=-1, keepdims=True)
        return fused.astype(np.float32)

    def fuse_predictions(
        self,
        pred_1d: BranchPrediction,
        pred_2d: BranchPrediction,
        raw_iq: Optional[np.ndarray] = None,
    ) -> FusionPrediction:
        """
        Fuses single-window BranchPrediction objects into a full ASTRA FusionPrediction.
        Optionally applies Stage 1 statistical prior gating if raw IQ is provided.
        """
        # 1. Validation of alignments
        validate_class_alignment(pred_1d.class_names, pred_2d.class_names)
        validate_source_alignment(
            pred_1d.source_signal_id, pred_1d.window_start, pred_1d.window_end,
            pred_2d.source_signal_id, pred_2d.window_start, pred_2d.window_end,
        )

        class_names = pred_1d.class_names
        p1 = np.asarray(pred_1d.probabilities, dtype=np.float32)
        p2 = np.asarray(pred_2d.probabilities, dtype=np.float32)

        # 2. Probability fusion
        p_fused = self.fuse_probabilities(p1, p2)

        # Apply Statistical Prior Gating (Stage 1 Enhancement)
        if raw_iq is not None and len(raw_iq) > 0:
            p_fused, _ = apply_statistical_prior_gating(p_fused, class_names, raw_iq)

        # 3. Branch agreement calculation
        pred_class_1d = pred_1d.predicted_class
        pred_class_2d = pred_2d.predicted_class
        branch_agreement = (pred_class_1d.strip().lower() == pred_class_2d.strip().lower())

        # 4. Fused winner & Top-K ranking
        sorted_indices = np.argsort(p_fused)[::-1]
        best_idx = int(sorted_indices[0])
        fused_class = class_names[best_idx]
        fused_confidence = float(p_fused[best_idx])

        # Top-K candidate list
        top_k_candidates: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indices[: self.top_k], start=1):
            top_k_candidates.append({
                "rank": rank,
                "class": class_names[idx],
                "probability": float(p_fused[idx]),
            })

        # 5. Margins, Entropy, and Status
        top1_p, top2_p, margin = self.confidence_calc.compute_margin(p_fused)
        entropy = self.confidence_calc.compute_entropy(p_fused)
        is_unknown = (fused_class.strip().lower() == "unknown")

        status, unknown_reason = self.confidence_calc.determine_status(
            predicted_class=fused_class,
            confidence=fused_confidence,
            confidence_margin=margin,
            branch_agreement=branch_agreement,
            entropy=entropy,
            is_unknown_class=is_unknown,
        )

        # 6. Branch evidence dictionary
        branch_evidence = {
            "resnet1d": {
                "model_name": pred_1d.model_name,
                "predicted_class": pred_1d.predicted_class,
                "confidence": float(pred_1d.confidence),
                "top_k": pred_1d.top_k,
            },
            "spectrogram2d": {
                "model_name": pred_2d.model_name,
                "predicted_class": pred_2d.predicted_class,
                "confidence": float(pred_2d.confidence),
                "top_k": pred_2d.top_k,
            },
        }

        # 7. Class probabilities dictionary
        probs_dict = {name: float(p_fused[i]) for i, name in enumerate(class_names)}

        # 8. Model version tracking
        model_versions = {
            "resnet1d": pred_1d.model_version,
            "spectrogram2d": pred_2d.model_version,
            "fusion": "astra_weighted_fusion_v1.0",
        }

        # 9. Return structured FusionPrediction
        return FusionPrediction(
            predicted_class=fused_class,
            confidence=fused_confidence,
            status=status,
            unknown_reason=unknown_reason,
            branch_agreement=branch_agreement,
            confidence_margin=margin,
            probability_entropy=entropy,
            fusion_mode="weighted_probability",
            fusion_weights={"resnet1d": self.weight_1d, "spectrogram2d": self.weight_2d},
            top_k=top_k_candidates,
            probabilities=probs_dict,
            branch_evidence=branch_evidence,
            model_versions=model_versions,
            source_signal_id=pred_1d.source_signal_id or pred_2d.source_signal_id,
            window_start=pred_1d.window_start,
            window_end=pred_1d.window_end,
        )

    def fuse_predictions_batch(
        self,
        preds_1d: Sequence[BranchPrediction],
        preds_2d: Sequence[BranchPrediction],
    ) -> List[FusionPrediction]:
        """
        Fuses a list of matched predictions.
        """
        if len(preds_1d) != len(preds_2d):
            raise ValueError(f"Batch size mismatch: {len(preds_1d)} vs {len(preds_2d)}")

        return [self.fuse_predictions(p1, p2) for p1, p2 in zip(preds_1d, preds_2d)]
