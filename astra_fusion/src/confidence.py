"""
ASTRA Confidence, Margin, Entropy, and Status Assignment Engine.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class ConfidenceCalculator:
    """
    Computes confidence metrics, margin, normalized entropy, and ASTRA status tiers.
    """

    def __init__(
        self,
        confirmed_threshold: float = 0.85,
        estimated_threshold: float = 0.50,
        possible_threshold: float = 0.30,
        confirmed_margin: float = 0.20,
        entropy_threshold: float = 0.80,
    ):
        self.confirmed_threshold = confirmed_threshold
        self.estimated_threshold = estimated_threshold
        self.possible_threshold = possible_threshold
        self.confirmed_margin = confirmed_margin
        self.entropy_threshold = entropy_threshold

    @staticmethod
    def compute_entropy(probs: np.ndarray, epsilon: float = 1e-12) -> float:
        """
        Computes normalized Shannon entropy in the range [0.0, 1.0].
        0.0 = completely deterministic (single class probability = 1.0)
        1.0 = maximum uniform uncertainty
        """
        p = np.asarray(probs, dtype=np.float64)
        p = np.clip(p, epsilon, 1.0)
        num_classes = len(p)
        if num_classes <= 1:
            return 0.0

        raw_entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(num_classes)
        normalized_entropy = float(raw_entropy / (max_entropy + epsilon))
        return float(np.clip(normalized_entropy, 0.0, 1.0))

    @staticmethod
    def compute_margin(probs: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (top1_prob, top2_prob, confidence_margin).
        """
        p = np.asarray(probs, dtype=np.float64)
        sorted_p = np.sort(p)[::-1]
        top1 = float(sorted_p[0])
        top2 = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
        margin = float(top1 - top2)
        return top1, top2, margin

    def determine_status(
        self,
        predicted_class: str,
        confidence: float,
        confidence_margin: float,
        branch_agreement: bool,
        entropy: float,
        is_unknown_class: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """
        Assigns an ASTRA confidence status tier and identifies unknown reason.
        
        Statuses:
          - CONFIRMED: High confidence, high margin, and branch agreement.
          - ESTIMATED: Strong probability above estimated threshold.
          - POSSIBLE: Moderate probability above possible threshold.
          - UNKNOWN: Low confidence, high entropy, branch conflict, or explicit unknown class.
          
        Returns:
            Tuple[status_str, unknown_reason_str_or_none]
        """
        # Case A: Explicit model "Unknown" class
        if is_unknown_class or predicted_class.strip().lower() == "unknown":
            return "UNKNOWN", "explicit_unknown_class"

        # Case B: CONFIRMED
        if (
            branch_agreement
            and confidence >= self.confirmed_threshold
            and confidence_margin >= self.confirmed_margin
        ):
            return "CONFIRMED", None

        # Case C: ESTIMATED
        if confidence >= self.estimated_threshold:
            return "ESTIMATED", None

        # Case D: POSSIBLE
        if confidence >= self.possible_threshold:
            return "POSSIBLE", None

        # Case E: UNKNOWN (determine precise rationale)
        if not branch_agreement and confidence < self.estimated_threshold:
            return "UNKNOWN", "branch_conflict"
        elif entropy >= self.entropy_threshold:
            return "UNKNOWN", "high_uncertainty"
        else:
            return "UNKNOWN", "low_confidence"
