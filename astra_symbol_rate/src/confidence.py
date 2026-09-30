"""
ASTRA Symbol-Rate Confidence, Margin, and Status Tier Assignment.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

from .models import SymbolRateCandidate


class SymbolRateConfidenceCalculator:
    """
    Evaluates candidate probability margin, support count, and assigns ASTRA status.
    """

    def __init__(
        self,
        confirmed_threshold: float = 0.80,
        estimated_threshold: float = 0.50,
        possible_threshold: float = 0.28,
        confirmed_margin: float = 0.18,
        min_support_count_confirmed: int = 2,
    ):
        self.confirmed_threshold = confirmed_threshold
        self.estimated_threshold = estimated_threshold
        self.possible_threshold = possible_threshold
        self.confirmed_margin = confirmed_margin
        self.min_support_count_confirmed = min_support_count_confirmed

    def compute_margin(self, ranked_candidates: Sequence[SymbolRateCandidate]) -> Tuple[float, float, float]:
        """
        Returns (top1_score, top2_score, margin).
        """
        if not ranked_candidates:
            return 0.0, 0.0, 0.0

        top1 = float(ranked_candidates[0].score)
        top2 = float(ranked_candidates[1].score) if len(ranked_candidates) > 1 else 0.0
        margin = float(top1 - top2)
        return top1, top2, margin

    def determine_status(
        self,
        ranked_candidates: Sequence[SymbolRateCandidate],
    ) -> Tuple[str, Optional[str], float, float]:
        """
        Assigns ASTRA Status tier:
          - CONFIRMED: High score, clear margin, and multiple supporting DSP estimators.
          - ESTIMATED: Strong score above estimated threshold.
          - POSSIBLE: Moderate score above possible threshold.
          - UNKNOWN: Ambiguous, low score, or no candidates.
          
        Returns:
            Tuple[status, unknown_reason, confidence, margin]
        """
        if not ranked_candidates:
            return "UNKNOWN", "no_dsp_candidates", 0.0, 0.0

        top1, top2, margin = self.compute_margin(ranked_candidates)
        best_cand = ranked_candidates[0]
        support_count = len(best_cand.supported_by)

        # 1. CONFIRMED
        if (
            top1 >= self.confirmed_threshold
            and margin >= self.confirmed_margin
            and support_count >= self.min_support_count_confirmed
        ):
            return "CONFIRMED", None, top1, margin

        # 2. ESTIMATED
        if top1 >= self.estimated_threshold:
            return "ESTIMATED", None, top1, margin

        # 3. POSSIBLE
        if top1 >= self.possible_threshold:
            return "POSSIBLE", None, top1, margin

        # 4. UNKNOWN
        if margin < 0.05 and len(ranked_candidates) > 1:
            return "UNKNOWN", "severe_candidate_disagreement", top1, margin
        else:
            return "UNKNOWN", "low_ranker_confidence", top1, margin


def compute_confidence_and_status(
    candidates: Sequence[SymbolRateCandidate],
    evidence: Optional[Any] = None,
    config: Optional[Any] = None,
) -> Tuple[float, str, float, Optional[str]]:
    """
    Module-level helper to compute confidence, status, margin, and unknown reason.
    Returns (confidence, status, margin, unknown_reason).
    """
    calc = SymbolRateConfidenceCalculator()
    status, unk_reason, conf, margin = calc.determine_status(candidates)
    return conf, status, margin, unk_reason

