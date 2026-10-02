"""
ASTRA Modulation Intelligence V2 — Calibrated Fusion Engine.

Implements:
1. Calibrated multi-branch weighted probability fusion:
   P_fused = w1 * P_1D + w2 * P_2D
2. Secondary evidence injection from Random Forest family classifier:
   P_final(c) = normalize(P_fused(c) * (1 + beta * P_RF(family(c))))
3. Explicit uncertainty quantification:
   - Shannon entropy
   - Confidence margin (P_top1 - P_top2)
   - Branch agreement score (cosine similarity between P_1D and P_2D)
   - Branch disagreement score (1 - agreement)
   - UNKNOWN / Non-Target rejection score
4. Strict compliance with V2 Output Contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from astra_modulation_v2.class_schema import (
    CLASS_SCHEMA_VERSION,
    MODULATION_CLASSES_V2,
    NUM_CLASSES_V2,
    get_class_name,
    get_class_index,
    assert_runtime_class_order,
)
from astra_modulation_v2.dataset_builder import IQPreprocessorV2, iq_to_tensor_1d, iq_to_spectrogram_2d
from astra_modulation_v2.models.resnet1d import ResNet1DV2
from astra_modulation_v2.models.spectrogram_cnn import SpectrogramCNN2DV2
from astra_modulation_v2.rf_support import RFFamilySupportAdapter


class CalibratedFusionEngineV2:
    """
    Production-grade Modulation Intelligence V2 Fusion Engine.
    Combines ResNet-1D, Spectrogram CNN-2D, and RF family evidence.
    """

    def __init__(
        self,
        checkpoint_1d: Optional[str] = None,
        checkpoint_2d: Optional[str] = None,
        weight_1d: float = 0.90,
        weight_2d: float = 0.10,
        rf_beta: float = 0.0,
        temperature_1d: float = 1.5,
        temperature_2d: float = 1.5,
        top_k: int = 5,
        device: Optional[str] = None,
    ):
        self.root = Path(__file__).resolve().parent.parent
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Branch weights
        tot_w = weight_1d + weight_2d
        self.w_1d = weight_1d / max(tot_w, 1e-6)
        self.w_2d = weight_2d / max(tot_w, 1e-6)
        self.rf_beta = rf_beta
        
        self.temp_1d = max(temperature_1d, 0.05)
        self.temp_2d = max(temperature_2d, 0.05)
        self.top_k = top_k
        self.classes = list(MODULATION_CLASSES_V2)

        # Preprocessor & RF adapter
        self.preprocessor = IQPreprocessorV2(remove_dc=True, normalize_rms=True)
        self.rf_adapter = RFFamilySupportAdapter()

        # Models
        self.model_1d: Optional[ResNet1DV2] = None
        self.model_2d: Optional[SpectrogramCNN2DV2] = None

        ckpt_1d_path = checkpoint_1d or str(self.root / "checkpoints" / "astra_resnet1d_v2.pt")
        ckpt_2d_path = checkpoint_2d or str(self.root / "checkpoints" / "astra_spectrogram_cnn_v2.pt")

        self._load_checkpoints(ckpt_1d_path, ckpt_2d_path)

    def _load_checkpoints(self, path_1d: str, path_2d: str):
        if os.path.exists(path_1d):
            self.model_1d = ResNet1DV2(in_channels=2, num_classes=NUM_CLASSES_V2).to(self.device)
            ckpt = torch.load(path_1d, map_location=self.device)
            state = ckpt.get("state_dict", ckpt)
            # Adapt weights if checkpoint class count differs from schema
            model_dict = self.model_1d.state_dict()
            for k, v in state.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                elif k in model_dict and "classifier" in k:
                    # Partial copy for classifier head
                    min_classes = min(model_dict[k].shape[0], v.shape[0])
                    model_dict[k][:min_classes] = v[:min_classes]
            self.model_1d.load_state_dict(model_dict)
            self.model_1d.eval()

        if os.path.exists(path_2d):
            self.model_2d = SpectrogramCNN2DV2(in_channels=1, num_classes=NUM_CLASSES_V2).to(self.device)
            ckpt = torch.load(path_2d, map_location=self.device)
            state = ckpt.get("state_dict", ckpt)
            model_dict = self.model_2d.state_dict()
            for k, v in state.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                elif k in model_dict and "classifier" in k:
                    min_classes = min(model_dict[k].shape[0], v.shape[0])
                    model_dict[k][:min_classes] = v[:min_classes]
            self.model_2d.load_state_dict(model_dict)
            self.model_2d.eval()

    def classify(
        self,
        iq_samples: np.ndarray,
        sample_rate: float = 192000.0,
        window_size: int = 2048,
    ) -> Dict[str, Any]:
        """
        Classifies input raw IQ array using V2 fused pipeline.
        Returns full V2 output contract.
        """
        iq_clean = self.preprocessor.process(iq_samples)
        if len(iq_clean) < window_size:
            iq_window = np.pad(iq_clean, (0, window_size - len(iq_clean)), mode="constant")
        else:
            iq_window = iq_clean[:window_size]

        # 1. 1D Inference
        p_1d = np.ones(NUM_CLASSES_V2, dtype=np.float32) / NUM_CLASSES_V2
        if self.model_1d is not None:
            t_1d = iq_to_tensor_1d(iq_window).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_1d, _ = self.model_1d(t_1d)
                logits_1d = logits_1d / self.temp_1d
                p_1d = torch.softmax(logits_1d, dim=-1).cpu().numpy()[0]

        # 2. 2D Inference
        p_2d = np.ones(NUM_CLASSES_V2, dtype=np.float32) / NUM_CLASSES_V2
        if self.model_2d is not None:
            t_2d = iq_to_spectrogram_2d(iq_window).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_2d, _ = self.model_2d(t_2d)
                logits_2d = logits_2d / self.temp_2d
                p_2d = torch.softmax(logits_2d, dim=-1).cpu().numpy()[0]

        # 3. Base Calibrated Probability Fusion
        p_fused = self.w_1d * p_1d + self.w_2d * p_2d

        # 4. RF Family Support Evidence Injection
        rf_family_probs = self.rf_adapter.get_family_probabilities(iq_samples, sample_rate=sample_rate)
        rf_support_vec = self.rf_adapter.get_class_support_vector(rf_family_probs)

        # Modulate fused probabilities with RF evidence:
        p_final = p_fused * (1.0 + self.rf_beta * rf_support_vec)
        
        # Uncertainty & Out-of-Distribution injection for noise / non-target signals:
        unk_idx = get_class_index("UNKNOWN")
        if rf_family_probs.get("UNKNOWN", 0.0) > 0.40:
            p_final[unk_idx] += rf_family_probs["UNKNOWN"] * 0.50

        sum_p = np.sum(p_final)
        if sum_p > 0:
            p_final = p_final / sum_p
        else:
            p_final = p_fused

        # 5. Agreement & Uncertainty Metrics
        norm_1d = np.linalg.norm(p_1d)
        norm_2d = np.linalg.norm(p_2d)
        if norm_1d > 0 and norm_2d > 0:
            agreement = float(np.dot(p_1d, p_2d) / (norm_1d * norm_2d))
        else:
            agreement = 1.0
        disagreement = float(max(0.0, 1.0 - agreement))

        # Shannon Entropy
        safe_p = np.clip(p_final, 1e-12, 1.0)
        entropy = float(-np.sum(safe_p * np.log2(safe_p)) / np.log2(NUM_CLASSES_V2))

        # Sorted Candidates & Margin
        sorted_indices = np.argsort(p_final)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1])
        margin = float(p_final[top1_idx] - p_final[top2_idx])

        top_k_candidates = []
        for rank, idx in enumerate(sorted_indices[: self.top_k]):
            top_k_candidates.append({
                "rank": rank + 1,
                "modulation": self.classes[idx],
                "confidence": float(round(p_final[idx], 4)),
                "prob_1d": float(round(p_1d[idx], 4)),
                "prob_2d": float(round(p_2d[idx], 4)),
            })

        # UNKNOWN / Rejection Score
        unknown_idx = get_class_index("UNKNOWN")
        unknown_score = float(round(p_final[unknown_idx], 4))

        # Class Probs Dict
        class_probs = {self.classes[i]: float(round(p_final[i], 4)) for i in range(NUM_CLASSES_V2)}

        return {
            "predicted_modulation": self.classes[top1_idx],
            "top1_confidence": float(round(p_final[top1_idx], 4)),
            "top_k_candidates": top_k_candidates,
            "class_probs": class_probs,
            "prob_1d": {self.classes[i]: float(round(p_1d[i], 4)) for i in range(NUM_CLASSES_V2)},
            "prob_2d": {self.classes[i]: float(round(p_2d[i], 4)) for i in range(NUM_CLASSES_V2)},
            "rf_family_evidence": rf_family_probs,
            "agreement_score": float(round(agreement, 4)),
            "disagreement_score": float(round(disagreement, 4)),
            "entropy": float(round(entropy, 4)),
            "margin": float(round(margin, 4)),
            "unknown_score": unknown_score,
            "schema_version": CLASS_SCHEMA_VERSION,
        }
