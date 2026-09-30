"""
scorer.py
Candidate scoring engine for ASTRA Stage 11.
Evaluates single or batch candidates using trained XGBoost models with rule-based fallback.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import xgboost as xgb

from .models import PipelineRankingResult, PipelinePathCandidate
from .feature_builder import PipelineFeatureBuilder
from .feature_schema import PIPELINE_FEATURE_COLUMNS, PIPELINE_FEATURE_SCHEMA_VERSION
from .calibration import ScoreCalibrator
from .ranking import rank_pipeline_candidates
from .feature_importance import extract_xgboost_feature_importance


class PipelineScorer:
    """
    Inference scoring and ranking engine for candidate signal-recovery pipelines.
    """

    def __init__(
        self,
        model: Optional[xgb.XGBClassifier] = None,
        calibrator: Optional[ScoreCalibrator] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.calibrator = calibrator
        self.config = config or {}
        self.feature_builder = PipelineFeatureBuilder()
        self.model_version = self.config.get("model_version", "astra_pipeline_xgb_v1")
        self.feature_schema_version = PIPELINE_FEATURE_SCHEMA_VERSION
        self._feature_importance: Optional[Dict[str, float]] = None

        if self.model is not None:
            try:
                self._feature_importance = extract_xgboost_feature_importance(self.model)
            except Exception:
                self._feature_importance = None

    def score_candidate(self, candidate: Any) -> float:
        """Score a single candidate pipeline. Returns score in [0.0, 1.0]."""
        results = self.score_batch([candidate])
        return float(results[0]) if len(results) > 0 else 0.0

    def score_batch(self, candidates: List[Any]) -> np.ndarray:
        """
        Score a batch of candidates. Uses XGBoost model if available;
        otherwise falls back to rule-based Stage 10 overall validation score.
        """
        if not candidates:
            return np.empty((0,), dtype=np.float32)

        if self.model is not None:
            # XGBoost batch inference
            X = self.feature_builder.extract_feature_matrix(candidates)
            raw_scores = self.model.predict_proba(X)[:, 1]
            return np.asarray(raw_scores, dtype=np.float32)

        # Fallback: Stage 10 rule-based score
        fallback_scores = []
        for c in candidates:
            if hasattr(c, "stage10_validation_metrics"):
                s10 = c.stage10_validation_metrics or {}
                sc = s10.get("overall_validation_score", 0.5)
            elif isinstance(c, dict):
                s10 = c.get("stage10_validation_metrics", {}) or c.get("validation_features", {}) or {}
                sc = s10.get("overall_validation_score", c.get("overall_validation_score", 0.5))
            else:
                sc = 0.5
            fallback_scores.append(float(sc))
        return np.array(fallback_scores, dtype=np.float32)

    def rank_candidates(
        self,
        candidates: List[Any],
        signal_id: str = "signal_0",
        top_k: Optional[int] = None,
    ) -> PipelineRankingResult:
        """
        Score, calibrate, and rank competing candidates for a given capture.
        """
        k = top_k or self.config.get("top_k", 3)
        if not candidates:
            return rank_pipeline_candidates(
                [], np.array([]), None, None, None, k, signal_id, self.model_version
            )

        fallback_used = (self.model is None)
        scores = self.score_batch(candidates)

        if self.calibrator is not None and self.calibrator.is_fitted:
            calibrated = self.calibrator.calibrate(scores)
        else:
            calibrated = scores

        # Extract feature dicts for explanation generation
        feat_dicts = [self.feature_builder.extract_features(c) for c in candidates]

        return rank_pipeline_candidates(
            candidates=candidates,
            scores=scores,
            calibrated_probs=calibrated,
            candidate_features_list=feat_dicts,
            feature_importance=self._feature_importance,
            top_k=k,
            signal_id=signal_id,
            model_version=self.model_version,
            fallback_used=fallback_used,
        )
