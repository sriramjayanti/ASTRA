"""
ASTRA Random Forest Support Engine Production Inference API.
Provides broad signal-family classification, signal-quality evidence, and batch inference.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import yaml

from .checkpoint import load_random_forest_bundle
from .family_classifier import BroadFamilyClassifier
from .feature_extractor_adapter import extract_dsp_feature_dict
from .feature_schema import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    IncompatibleFeatureSchemaError,
    validate_feature_matrix,
    validate_feature_vector,
)
from .preprocessing import FeaturePreprocessor
from .quality_classifier import SignalQualityClassifier

logger = logging.getLogger("astra_random_forest.inference")


class RandomForestSupportEngine:
    """
    Main ASTRA Random Forest Support Engine.
    Provides explainable broad signal-family and channel quality evidence.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.model_version = "astra_rf_support_v1.0"
        self.feature_schema_version = FEATURE_SCHEMA_VERSION
        self.feature_columns = list(FEATURE_COLUMNS)

        self.family_model = BroadFamilyClassifier()
        self.quality_model = SignalQualityClassifier()
        self.preprocessor = FeaturePreprocessor()
        self.metadata: Dict[str, Any] = {}

        # 1. Resolve config
        self.config: Dict[str, Any] = {}
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            default_cfg = Path(__file__).resolve().parent.parent / "configs" / "random_forest_config.yaml"
            if default_cfg.exists():
                with open(default_cfg, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}

        # 2. Resolve Checkpoint
        resolved_ckpt = checkpoint_path
        if resolved_ckpt is None:
            default_ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / "random_forest_support.joblib"
            if default_ckpt.exists():
                resolved_ckpt = str(default_ckpt)

        if resolved_ckpt is not None and os.path.exists(resolved_ckpt):
            try:
                f_model, q_model, prep, meta = load_random_forest_bundle(resolved_ckpt)
                self.family_model = f_model
                self.quality_model = q_model
                self.preprocessor = prep
                self.metadata = meta
                logger.info("Loaded Random Forest support model from: %s", resolved_ckpt)
            except Exception as e:
                logger.warning("Failed to load checkpoint from %s: %s. Using untracked model.", resolved_ckpt, e)

    def _prepare_feature_vector(
        self,
        features: Union[Dict[str, Any], np.ndarray],
        sample_rate_hz: float = 192000.0,
    ) -> np.ndarray:
        """
        Converts IQ samples or feature dict into an imputed [1, 36] feature vector.
        """
        if isinstance(features, np.ndarray):
            if np.iscomplexobj(features) or (features.ndim == 1 and len(features) > len(FEATURE_COLUMNS)):
                # Raw IQ signal passed in -> extract DSP features
                feat_dict = extract_dsp_feature_dict(features, sample_rate_hz=sample_rate_hz)
                raw_row = validate_feature_vector(feat_dict)
                raw_mat = np.asarray([raw_row], dtype=np.float32)
            else:
                raw_mat = validate_feature_matrix(features)
        elif isinstance(features, dict):
            raw_row = validate_feature_vector(features)
            raw_mat = np.asarray([raw_row], dtype=np.float32)
        else:
            raise TypeError("Input features must be a dict of DSP features or a numpy array.")

        return self.preprocessor.transform(raw_mat)

    def predict_family(
        self,
        features: Union[Dict[str, Any], np.ndarray],
        sample_rate_hz: float = 192000.0,
    ) -> Dict[str, Any]:
        """
        Predicts broad signal family (FSK, PSK, QAM, UNKNOWN).
        """
        X = self._prepare_feature_vector(features, sample_rate_hz=sample_rate_hz)
        probs = self.family_model.predict_proba(X)[0]
        classes = self.family_model.classes

        prob_dict = {cls_name: float(round(probs[i], 4)) for i, cls_name in enumerate(classes)}
        
        # Sort descending
        sorted_items = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        top1_cls, top1_prob = sorted_items[0]
        top2_prob = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        margin = float(round(top1_prob - top2_prob, 4))

        status = "CONFIRMED" if top1_prob >= 0.80 and margin >= 0.20 else ("ESTIMATED" if top1_prob >= 0.50 else "POSSIBLE")

        return {
            "predicted_family": top1_cls,
            "confidence": top1_prob,
            "status": status,
            "confidence_margin": margin,
            "probabilities": prob_dict,
            "model_version": self.model_version,
            "feature_schema": self.feature_schema_version,
        }

    def predict_quality(
        self,
        features: Union[Dict[str, Any], np.ndarray],
        sample_rate_hz: float = 192000.0,
    ) -> Dict[str, Any]:
        """
        Predicts independent channel quality and corruption probabilities.
        """
        X = self._prepare_feature_vector(features, sample_rate_hz=sample_rate_hz)
        probs_dict = self.quality_model.predict_proba(X)

        quality_evidence = {}
        for lbl, p_arr in probs_dict.items():
            prob_val = float(round(float(p_arr[0]), 4))
            quality_evidence[lbl] = {
                "probability": prob_val,
                "detected": bool(prob_val >= 0.50),
            }

        return {
            "quality_evidence": quality_evidence,
            "model_version": self.model_version,
        }

    def predict_all(
        self,
        features: Union[Dict[str, Any], np.ndarray],
        sample_rate_hz: float = 192000.0,
    ) -> Dict[str, Any]:
        """
        Runs both Broad Family and Signal Quality models, returning unified supporting evidence.
        """
        family_res = self.predict_family(features, sample_rate_hz=sample_rate_hz)
        quality_res = self.predict_quality(features, sample_rate_hz=sample_rate_hz)

        # Top supporting features
        top_importances = sorted(
            self.family_model.get_feature_importances().items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        return {
            "model": self.model_version,
            "family": family_res["predicted_family"],
            "confidence": family_res["confidence"],
            "status": family_res["status"],
            "confidence_margin": family_res["confidence_margin"],
            "probabilities": family_res["probabilities"],
            "quality": {
                "low_snr_probability": quality_res["quality_evidence"].get("low_snr", {}).get("probability", 0.0),
                "clipping_probability": quality_res["quality_evidence"].get("clipped", {}).get("probability", 0.0),
                "multipath_probability": quality_res["quality_evidence"].get("multipath", {}).get("probability", 0.0),
                "cfo_affected_probability": quality_res["quality_evidence"].get("cfo_affected", {}).get("probability", 0.0),
            },
            "important_evidence": [
                {"feature": feat, "importance": float(round(imp, 4))} for feat, imp in top_importances
            ],
            "feature_schema": self.feature_schema_version,
        }

    def predict_batch(
        self,
        features_list: Sequence[Union[Dict[str, Any], np.ndarray]],
        sample_rate_hz: float = 192000.0,
    ) -> List[Dict[str, Any]]:
        """
        Performs batch inference on a collection of signals/feature vectors.
        """
        results = []
        for item in features_list:
            res = self.predict_all(item, sample_rate_hz=sample_rate_hz)
            results.append(res)
        return results
