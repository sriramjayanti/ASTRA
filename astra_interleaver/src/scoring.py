"""
scoring.py
Rule-based structural scoring and status assignment for deinterleaver candidates.
Evaluates weak pre-FEC evidence (periodicity, autocorrelation, divisibility, run lengths, family priors).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .models import StructuralFeatures, InterleaverStatus, InterleaverCandidateResult


class CandidateScorer:
    """
    Transparent, deterministic rule-based structural scoring engine.
    Does NOT attempt premature definitive classification; preserves candidate diversity for Stage 9 FEC.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        scoring_cfg = cfg.get("scoring", {})
        self.periodicity_weight = float(scoring_cfg.get("periodicity_weight", 1.0))
        self.autocorrelation_weight = float(scoring_cfg.get("autocorrelation_weight", 1.0))
        self.entropy_weight = float(scoring_cfg.get("entropy_weight", 0.3))
        self.run_length_weight = float(scoring_cfg.get("run_length_weight", 0.4))
        self.bit_balance_weight = float(scoring_cfg.get("bit_balance_weight", 0.2))
        self.family_prior_weight = float(scoring_cfg.get("family_prior_weight", 0.3))
        self.divisibility_weight = float(scoring_cfg.get("divisibility_weight", 0.5))
        
        self.family_priors = cfg.get("family_priors", {
            "identity": 1.0,
            "block": 0.85,
            "convolutional": 0.70,
            "helical": 0.50,
            "pseudo_random": 0.40
        })

    def score_candidate(
        self,
        features: StructuralFeatures,
        family: str,
        success: bool = True
    ) -> Tuple[float, float, float, float, float, InterleaverStatus]:
        """
        Computes sub-scores and overall interleaver candidate score.
        
        Returns:
            (overall_score, periodicity_score, autocorrelation_score, entropy_score, run_length_score, status)
        """
        if not success:
            return 0.0, 0.0, 0.0, 0.0, 0.0, InterleaverStatus.INTERLEAVER_INVALID
            
        # 1. Periodicity and autocorrelation scores
        periodicity_sc = features.periodicity_score
        autocorr_sc = float(np.clip(features.autocorrelation_peak, 0.0, 1.0))
        
        # 2. Entropy score (reward reasonable entropy [0.8, 1.0] without punishing compressed data)
        if features.binary_entropy >= 0.75:
            entropy_sc = 1.0
        else:
            entropy_sc = float(np.clip(features.binary_entropy / 0.75, 0.0, 1.0))
            
        # 3. Run-length score (reward balanced mean run length ~ 1.5 - 3.5 typical of coded bits)
        mean_run = features.mean_run_length
        if 1.2 <= mean_run <= 4.0:
            run_sc = 1.0
        elif mean_run < 1.2:
            run_sc = 0.6
        else:
            run_sc = float(np.clip(4.0 / (mean_run + 1e-6), 0.1, 1.0))
            
        # 4. Divisibility / Remainder penalty
        rem_fraction = features.remainder_fraction
        divisibility_penalty = rem_fraction * self.divisibility_weight
        
        # 5. Family prior
        prior = float(self.family_priors.get(family, 0.5))
        
        # Weighted combination
        raw_score = (
            self.periodicity_weight * periodicity_sc +
            self.autocorrelation_weight * autocorr_sc +
            self.entropy_weight * entropy_sc +
            self.run_length_weight * run_sc +
            self.family_prior_weight * prior -
            divisibility_penalty
        )
        
        max_possible = (
            self.periodicity_weight +
            self.autocorrelation_weight +
            self.entropy_weight +
            self.run_length_weight +
            self.family_prior_weight
        )
        
        # 6. Known sync word / frame marker correlation boost
        sync_corr = features.known_sync_correlation
        if sync_corr > 0.85:
            raw_score += 2.0 * sync_corr
            max_possible += 1.0

        normalized_score = float(np.clip(raw_score / (max_possible + 1e-6), 0.01, 1.0))
        
        # Status determination
        if normalized_score >= 0.35:
            status = InterleaverStatus.INTERLEAVER_PLAUSIBLE
        else:
            status = InterleaverStatus.INTERLEAVER_WEAK
            
        return normalized_score, periodicity_sc, autocorr_sc, entropy_sc, run_sc, status
