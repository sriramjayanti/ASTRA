"""
inference.py
Top-level pipeline inference interface for ASTRA Stage 11 — Pipeline Scoring Model.
Loads configured model checkpoints and provides seamless ranking for upstream candidates.
"""

from typing import Any, Dict, List, Optional, Union
import os
import yaml

from .models import PipelineRankingResult, PipelinePathCandidate
from .scorer import PipelineScorer
from .checkpoint import load_pipeline_model_package


class PipelineScoringEngine:
    """
    Production entrypoint for Stage 11 candidate pipeline scoring.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.config = self._load_config(config_path)
        self.scorer = self._init_scorer(checkpoint_dir)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None or not os.path.exists(config_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "configs", "pipeline_scorer_config.yaml")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("pipeline_scorer", {})
        return {}

    def _init_scorer(self, checkpoint_dir: Optional[str]) -> PipelineScorer:
        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            try:
                model, calib, meta = load_pipeline_model_package(checkpoint_dir)
                return PipelineScorer(model=model, calibrator=calib, config=self.config)
            except Exception:
                pass
        # Initialize scorer with rule-based fallback if no checkpoint provided
        return PipelineScorer(model=None, calibrator=None, config=self.config)

    def rank(
        self,
        candidates: List[Any],
        signal_id: str = "signal_0",
        top_k: Optional[int] = None,
    ) -> PipelineRankingResult:
        """
        Rank a collection of competing pipeline candidates for a single signal capture.
        """
        return self.scorer.rank_candidates(candidates, signal_id=signal_id, top_k=top_k)
