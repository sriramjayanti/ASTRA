"""
ASTRA Production Multi-Branch Modulation Fusion Engine.
Main inference entry point unifying 1D ResNet and 2D Spectrogram CNN evidence.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import yaml

from .adapters import (
    BaseBranchAdapter,
    ResNet1DAdapter,
    Spectrogram2DAdapter,
    preprocess_iq,
)
from .calibration import TemperatureScaler
from .confidence import ConfidenceCalculator
from .learned_fusion import LearnedFusionEngine, LearnedFusionMLP
from .models import (
    BranchPrediction,
    CandidateItem,
    ClassMappingMismatchError,
    FusionPrediction,
    SourceAlignmentError,
)
from .validation import validate_class_alignment
from .weighted_fusion import WeightedProbabilityFusion


class ASTRAFusionEngine:
    """
    ASTRA Multi-Branch Modulation Fusion Engine.
    
    Orchestrates:
      1. Single IQ Window Ingestion
      2. 1D ResNet Inference (Time-Domain Raw IQ)
      3. 2D Spectrogram CNN Inference (Time-Frequency STFT)
      4. Dynamic Probability / Learned Feature Fusion
      5. Top-K Candidate Generation & ASTRA Confidence Statusing
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        adapter_1d: Optional[BaseBranchAdapter] = None,
        adapter_2d: Optional[BaseBranchAdapter] = None,
        mode: str = "weighted_probability",
        weight_1d: float = 0.70,
        weight_2d: float = 0.30,
        top_k: int = 3,
        thresholds: Optional[Dict[str, float]] = None,
        learned_model: Optional[LearnedFusionMLP] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.config: Dict[str, Any] = {}

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            default_cfg_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "fusion_config.yaml"
            )
            if os.path.exists(default_cfg_path):
                with open(default_cfg_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}

        f_cfg = self.config.get("fusion", {})
        self.mode = mode if mode != "weighted_probability" else f_cfg.get("mode", mode)
        self.top_k = top_k if top_k != 3 else f_cfg.get("top_k", top_k)

        # Thresholds
        t_cfg = f_cfg.get("thresholds", thresholds or {})
        self.confidence_calc = ConfidenceCalculator(
            confirmed_threshold=t_cfg.get("confirmed", 0.85),
            estimated_threshold=t_cfg.get("estimated", 0.50),
            possible_threshold=t_cfg.get("possible", 0.30),
            confirmed_margin=t_cfg.get("confirmed_margin", 0.20),
        )

        # Initialize Branch Adapters
        b_cfg = f_cfg.get("branches", {})
        r1d_cfg = b_cfg.get("resnet1d", {})
        s2d_cfg = b_cfg.get("spectrogram2d", {})

        w1 = r1d_cfg.get("weight", weight_1d)
        w2 = s2d_cfg.get("weight", weight_2d)

        self.weighted_engine = WeightedProbabilityFusion(
            weight_1d=w1,
            weight_2d=w2,
            top_k=self.top_k,
            confidence_calc=self.confidence_calc,
        )

        if adapter_1d is not None:
            self.adapter_1d = adapter_1d
        else:
            ckpt_1d = r1d_cfg.get("checkpoint_path", "checkpoints/astra_resnet1d_v2.pt")
            self.adapter_1d = ResNet1DAdapter(
                checkpoint_path=ckpt_1d,
                model_name=r1d_cfg.get("model_name", "astra_resnet1d_v2"),
                model_version=r1d_cfg.get("model_version", "2.0.0"),
                temperature=r1d_cfg.get("temperature", 1.0),
                device=self.device,
            )

        if adapter_2d is not None:
            self.adapter_2d = adapter_2d
        else:
            ckpt_2d = s2d_cfg.get("checkpoint_path", "checkpoints/astra_spectrogram_cnn_v2.pt")
            self.adapter_2d = Spectrogram2DAdapter(
                checkpoint_path=ckpt_2d,
                model_name=s2d_cfg.get("model_name", "astra_spectrogram2d_v2"),
                model_version=s2d_cfg.get("model_version", "2.0.0"),
                temperature=s2d_cfg.get("temperature", 1.0),
                device=self.device,
            )

        # Enforce class alphabet alignment on initialization
        validate_class_alignment(self.adapter_1d.class_names, self.adapter_2d.class_names)
        self.class_names = list(self.adapter_1d.class_names)

        # Mode B Learned Engine (Optional)
        self.learned_engine: Optional[LearnedFusionEngine] = None
        if learned_model is not None:
            self.learned_engine = LearnedFusionEngine(
                model=learned_model,
                class_names=self.class_names,
                top_k=self.top_k,
                confidence_calc=self.confidence_calc,
                device=self.device,
            )

    def set_weights(self, weight_1d: float, weight_2d: float) -> None:
        """Sets the convex weights for Mode A fusion."""
        self.weighted_engine.set_weights(weight_1d, weight_2d)

    def predict_1d_only(
        self,
        iq_window: Union[np.ndarray, torch.Tensor],
        source_signal_id: Optional[str] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
    ) -> BranchPrediction:
        """Runs inference strictly on the 1D ResNet branch."""
        return self.adapter_1d.predict_window(
            iq_window,
            source_signal_id=source_signal_id,
            window_start=window_start,
            window_end=window_end,
        )

    def predict_2d_only(
        self,
        iq_window: Union[np.ndarray, torch.Tensor],
        source_signal_id: Optional[str] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
    ) -> BranchPrediction:
        """Runs inference strictly on the 2D Spectrogram CNN branch."""
        return self.adapter_2d.predict_window(
            iq_window,
            source_signal_id=source_signal_id,
            window_start=window_start,
            window_end=window_end,
        )

    def fuse_predictions(
        self,
        pred_1d: BranchPrediction,
        pred_2d: BranchPrediction,
    ) -> FusionPrediction:
        """Fuses precomputed 1D and 2D BranchPredictions."""
        if self.mode == "learned" and self.learned_engine is not None:
            return self.learned_engine.fuse_predictions(pred_1d, pred_2d)
        return self.weighted_engine.fuse_predictions(pred_1d, pred_2d)

    def fuse_predictions_batch(
        self,
        preds_1d: Sequence[BranchPrediction],
        preds_2d: Sequence[BranchPrediction],
    ) -> List[FusionPrediction]:
        """Fuses a batch of precomputed BranchPredictions."""
        return [self.fuse_predictions(p1, p2) for p1, p2 in zip(preds_1d, preds_2d)]

    def predict(
        self,
        iq_window: Union[np.ndarray, torch.Tensor],
        source_signal_id: Optional[str] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
    ) -> FusionPrediction:
        """
        Unified Single Window Pipeline:
        Accepts ONE physical IQ window and executes concurrent 1D + 2D evaluation and fusion.
        """
        # Execute 1D Branch
        p1 = self.adapter_1d.predict_window(
            iq_window,
            source_signal_id=source_signal_id,
            window_start=window_start,
            window_end=window_end,
        )

        # Execute 2D Branch on the SAME underlying IQ window
        p2 = self.adapter_2d.predict_window(
            iq_window,
            source_signal_id=source_signal_id,
            window_start=window_start,
            window_end=window_end,
        )

        # Fuse evidence
        return self.fuse_predictions(p1, p2)

    def predict_fused(
        self,
        iq_window: Union[np.ndarray, torch.Tensor],
        source_signal_id: Optional[str] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
    ) -> FusionPrediction:
        """Alias for predict()."""
        return self.predict(
            iq_window,
            source_signal_id=source_signal_id,
            window_start=window_start,
            window_end=window_end,
        )

    def predict_batch(
        self,
        iq_batch: Union[np.ndarray, torch.Tensor],
        source_signal_ids: Optional[List[str]] = None,
        window_starts: Optional[List[int]] = None,
        window_ends: Optional[List[int]] = None,
    ) -> List[FusionPrediction]:
        """
        Batch Inference Pipeline:
        Executes vectorized batch inference across both branches and returns aligned FusionPredictions.
        """
        preds_1d = self.adapter_1d.predict_batch(
            iq_batch,
            source_signal_ids=source_signal_ids,
            window_starts=window_starts,
            window_ends=window_ends,
        )
        preds_2d = self.adapter_2d.predict_batch(
            iq_batch,
            source_signal_ids=source_signal_ids,
            window_starts=window_starts,
            window_ends=window_ends,
        )

        return self.fuse_predictions_batch(preds_1d, preds_2d)
