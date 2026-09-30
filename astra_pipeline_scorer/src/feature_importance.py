"""
feature_importance.py
Feature importance extractor and explainability utilities for ASTRA Stage 11.
Extracts XGBoost gain, split weight, permutation importance, and individual candidate rationales.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import xgboost as xgb
from .feature_schema import PIPELINE_FEATURE_COLUMNS


def extract_xgboost_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: Optional[List[str]] = None,
    importance_type: str = "gain",
) -> Dict[str, float]:
    """
    Extract normalized feature importance dictionary from trained XGBoost model.
    """
    if feature_names is None:
        feature_names = list(PIPELINE_FEATURE_COLUMNS)

    booster = model.get_booster()
    score_dict = booster.get_score(importance_type=importance_type)

    importance_map: Dict[str, float] = {col: 0.0 for col in feature_names}
    for k, v in score_dict.items():
        if k in importance_map:
            importance_map[k] = float(v)
        elif k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if idx < len(feature_names):
                importance_map[feature_names[idx]] = float(v)

    # Normalize sum to 1.0 if positive
    total = sum(importance_map.values())
    if total > 0:
        for k in importance_map:
            importance_map[k] = float(importance_map[k] / total)

    # Sort descending
    sorted_items = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
    return {k: round(v, 4) for k, v in sorted_items}


def explain_candidate_prediction(
    candidate_features: Dict[str, float],
    feature_importance: Dict[str, float],
    top_n: int = 4,
) -> Dict[str, Any]:
    """
    Construct human-readable key evidence drivers for a candidate prediction.
    """
    positives = []
    negatives = []

    # Check high-importance validation/demod signals
    crc_rate = candidate_features.get("crc_pass_rate", 0.0)
    if crc_rate >= 0.80:
        positives.append(f"CRC pass rate high ({crc_rate:.0%})")
    elif candidate_features.get("crc_available", 0.0) > 0 and crc_rate == 0.0:
        negatives.append("CRC checks failed across all frames")

    evm = candidate_features.get("evm_percent", 20.0)
    if evm < 12.0:
        positives.append(f"Low demodulation EVM ({evm:.1f}%)")
    elif evm > 25.0:
        negatives.append(f"High demodulation EVM ({evm:.1f}%)")

    synd_valid = candidate_features.get("syndrome_valid", 0.0)
    if synd_valid > 0:
        positives.append("FEC syndrome / parity checks valid")

    timing_lock = candidate_features.get("timing_lock_score", 0.0)
    if timing_lock >= 0.85:
        positives.append("Strong Gardner timing lock")

    rep_score = candidate_features.get("frame_periodicity_score", 0.0)
    if rep_score >= 0.50:
        positives.append("Periodic frame repetition detected")

    contradictions = candidate_features.get("contradiction_count", 0.0)
    if contradictions > 0:
        negatives.append(f"Recorded {int(contradictions)} cross-stage contradictions")

    return {
        "top_positive_evidence": positives[:top_n],
        "top_negative_evidence": negatives[:top_n],
    }
