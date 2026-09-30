"""
scoring.py
Rule-based scoring, family normalization, and FECStatus determination.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .models import FECStatus, DecoderResult, FECFamily


class FECScorer:
    """
    Transparent, deterministic rule-based candidate scorer for Stage 9.
    Evaluates multi-family decoder evidence without premature over-pruning.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        scoring_cfg = cfg.get("scoring", {})
        self.quality_weight = float(scoring_cfg.get("quality_weight", 1.0))
        self.path_metric_weight = float(scoring_cfg.get("path_metric_weight", 0.8))
        self.syndrome_weight = float(scoring_cfg.get("syndrome_weight", 1.0))
        self.parity_weight = float(scoring_cfg.get("parity_weight", 1.0))
        self.convergence_weight = float(scoring_cfg.get("convergence_weight", 1.0))
        self.family_prior_weight = float(scoring_cfg.get("family_prior_weight", 0.3))
        
        self.family_priors = {
            "none": 0.50,
            "convolutional": 0.80,
            "reed_solomon": 0.75,
            "concatenated": 0.70,
            "ldpc": 0.75
        }

    def score_candidate(
        self,
        decoder_result: DecoderResult,
        norm_metrics: Dict[str, Any],
        family: str
    ) -> Tuple[float, float, FECStatus]:
        """
        Computes FEC quality score, structural support score, and assigns candidate status.
        
        Returns:
            (fec_quality_score, structural_score, status)
        """
        if not decoder_result.success:
            # Failed decoding
            raw_score = 0.05
            return raw_score, 0.0, FECStatus.FEC_FAILED
            
        syn_sc = float(norm_metrics.get("syndrome_score", 0.0))
        parity_sc = 1.0 if norm_metrics.get("parity_success", False) else 0.0
        conv_sc = 1.0 if norm_metrics.get("converged", False) else 0.0
        norm_pm = float(norm_metrics.get("normalized_path_metric", 0.0))
        pm_sc = float(np.clip(1.0 - norm_pm, 0.0, 1.0))
        prior_sc = float(self.family_priors.get(family, 0.5))
        
        # Weighted aggregate score
        raw_score = (
            self.syndrome_weight * syn_sc +
            self.parity_weight * parity_sc +
            self.convergence_weight * conv_sc +
            self.path_metric_weight * pm_sc +
            self.family_prior_weight * prior_sc
        )
        
        max_possible = (
            self.syndrome_weight +
            self.parity_weight +
            self.convergence_weight +
            self.path_metric_weight +
            self.family_prior_weight
        )
        
        fec_quality = float(np.clip(raw_score / (max_possible + 1e-6), 0.05, 1.0))
        structural_sc = syn_sc
        
        if parity_sc == 1.0 and conv_sc == 1.0 and fec_quality >= 0.65:
            status = FECStatus.FEC_PLAUSIBLE
        elif fec_quality >= 0.40:
            status = FECStatus.FEC_PLAUSIBLE
        else:
            status = FECStatus.FEC_WEAK
            
        return fec_quality, structural_sc, status
