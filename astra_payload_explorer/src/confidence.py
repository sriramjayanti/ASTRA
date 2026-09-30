"""
confidence.py
Multi-dimensional confidence scoring and provenance attribution for Stage 14.
"""

from typing import Dict, Any, Optional, List
import numpy as np


def compute_boundary_confidence(
    support_sources: Optional[List[str]] = None,
    profile_match: bool = True,
    entropy_transition: bool = True,
    stage13_supported: bool = True
) -> float:
    """
    Calculate confidence score for structural boundary estimation based on corroborating sources.
    """
    sources = support_sources or []
    score = 0.5
    if profile_match or "profile" in sources:
        score += 0.3
    if stage13_supported or "stage13" in sources:
        score += 0.15
    if entropy_transition or "entropy" in sources:
        score += 0.05
    return float(np.clip(score, 0.0, 1.0))


def compute_frame_quality_score(
    is_partial: bool = False,
    is_corrupted: bool = False,
    boundary_confidence: float = 1.0,
    stage13_confidence: float = 0.9,
    crc_passed: Optional[bool] = None
) -> float:
    """
    Calculate consolidated frame quality score.
    """
    score = 1.0

    if is_partial:
        score *= 0.5

    if is_corrupted or crc_passed is False:
        score *= 0.4
    elif crc_passed is True:
        score = min(1.0, score * 1.1)

    score = 0.5 * score + 0.3 * boundary_confidence + 0.2 * stage13_confidence
    return float(np.clip(score, 0.0, 1.0))
