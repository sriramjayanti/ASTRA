"""
validators.py
Validation rules and physical feasibility checks for candidate hypotheses.
"""

import math
from typing import Dict, Any, List, Optional, Tuple


def validate_physical_feasibility(
    symbol_rate_hz: float,
    sample_rate_hz: float,
    min_sps: float = 1.0,
    max_sps: float = 2000.0
) -> Tuple[bool, Optional[str], float]:
    """
    Validate that symbol rate and sample rate form a physically feasible pair.
    
    Returns:
        (is_valid, rejection_reason, sps)
    """
    if math.isnan(symbol_rate_hz) or math.isinf(symbol_rate_hz) or symbol_rate_hz <= 0:
        return False, f"Invalid symbol rate: {symbol_rate_hz}", 0.0
    
    if math.isnan(sample_rate_hz) or math.isinf(sample_rate_hz) or sample_rate_hz <= 0:
        return False, f"Invalid sample rate: {sample_rate_hz}", 0.0

    sps = float(sample_rate_hz) / float(symbol_rate_hz)

    if sps < min_sps:
        return False, f"Samples per symbol ({sps:.3f}) below minimum limit ({min_sps}) - Nyquist violation", sps

    if sps > max_sps:
        return False, f"Samples per symbol ({sps:.3f}) exceeds maximum feasible limit ({max_sps})", sps

    return True, None, sps


def sanitize_probability(val: Any, default: float = 0.0, eps: float = 1e-8) -> float:
    """Ensure a probability value is a valid float in [0.0, 1.0] without NaN or Inf."""
    if val is None:
        return default
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return max(eps, min(1.0, f_val))
    except (TypeError, ValueError):
        return default


def parse_modulation_input(modulation_prediction: Any, top_k_limit: int = 3) -> List[Dict[str, Any]]:
    """
    Parse and normalize various structured modulation prediction formats.
    Supported formats:
      - Dict with "top_k": [{"class": "QPSK", "probability": 0.8}, ...]
      - Dict with "top_candidates": [{"modulation": "QPSK", "confidence": 0.8}, ...]
      - Dict of class->prob: {"QPSK": 0.8, "8PSK": 0.12}
      - List of dicts/tuples: [("QPSK", 0.8), ...]
    """
    candidates = []

    if isinstance(modulation_prediction, dict):
        if "top_k" in modulation_prediction and isinstance(modulation_prediction["top_k"], list):
            for item in modulation_prediction["top_k"]:
                mod = item.get("class") or item.get("modulation") or item.get("name")
                prob = item.get("probability", item.get("score", item.get("confidence", 0.0)))
                if mod:
                    candidates.append({"modulation": str(mod), "probability": sanitize_probability(prob)})
        elif "top_candidates" in modulation_prediction and isinstance(modulation_prediction["top_candidates"], list):
            for item in modulation_prediction["top_candidates"]:
                mod = item.get("class") or item.get("modulation") or item.get("name")
                prob = item.get("probability", item.get("score", item.get("confidence", 0.0)))
                if mod:
                    candidates.append({"modulation": str(mod), "probability": sanitize_probability(prob)})
        elif "probabilities" in modulation_prediction and isinstance(modulation_prediction["probabilities"], dict):
            sorted_probs = sorted(modulation_prediction["probabilities"].items(), key=lambda x: x[1], reverse=True)
            for mod, prob in sorted_probs[:top_k_limit]:
                candidates.append({"modulation": str(mod), "probability": sanitize_probability(prob)})
        else:
            # Assume dict of class -> prob directly
            sorted_probs = sorted(modulation_prediction.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
            for mod, prob in sorted_probs[:top_k_limit]:
                if isinstance(prob, (int, float)):
                    candidates.append({"modulation": str(mod), "probability": sanitize_probability(prob)})

    elif isinstance(modulation_prediction, list):
        for item in modulation_prediction:
            if isinstance(item, dict):
                mod = item.get("class") or item.get("modulation") or item.get("name")
                prob = item.get("probability", item.get("score", item.get("confidence", 0.0)))
                if mod:
                    candidates.append({"modulation": str(mod), "probability": sanitize_probability(prob)})
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                candidates.append({"modulation": str(item[0]), "probability": sanitize_probability(item[1])})

    # Sort descending by probability and limit to top_k
    candidates.sort(key=lambda x: x["probability"], reverse=True)
    return candidates[:top_k_limit]


def parse_symbol_rate_input(symbol_rate_prediction: Any, top_k_limit: int = 3) -> List[Dict[str, Any]]:
    """
    Parse and normalize various structured symbol rate prediction formats.
    Supported formats:
      - Dict with "top_k": [{"symbol_rate_hz": 9600, "score": 0.85}, ...]
      - Dict with "candidates": [...]
      - List of dicts or numbers
    """
    candidates = []

    if isinstance(symbol_rate_prediction, dict):
        items = symbol_rate_prediction.get("top_k", symbol_rate_prediction.get("candidates", []))
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    rate = item.get("symbol_rate_hz", item.get("baud", item.get("rate", item.get("value", 0.0))))
                    score = item.get("score", item.get("confidence", item.get("probability", 0.5)))
                    candidates.append({"symbol_rate_hz": float(rate), "score": sanitize_probability(score, default=0.5)})
                elif isinstance(item, (int, float)):
                    candidates.append({"symbol_rate_hz": float(item), "score": 0.5})
        elif "symbol_rate_hz" in symbol_rate_prediction:
            # Single prediction dict
            rate = symbol_rate_prediction["symbol_rate_hz"]
            score = symbol_rate_prediction.get("confidence", symbol_rate_prediction.get("score", 0.8))
            candidates.append({"symbol_rate_hz": float(rate), "score": sanitize_probability(score, default=0.8)})

    elif isinstance(symbol_rate_prediction, list):
        for item in symbol_rate_prediction:
            if isinstance(item, dict):
                rate = item.get("symbol_rate_hz", item.get("baud", item.get("rate", item.get("value", 0.0))))
                score = item.get("score", item.get("confidence", item.get("probability", 0.5)))
                candidates.append({"symbol_rate_hz": float(rate), "score": sanitize_probability(score, default=0.5)})
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                candidates.append({"symbol_rate_hz": float(item[0]), "score": sanitize_probability(item[1])})
            elif isinstance(item, (int, float)):
                candidates.append({"symbol_rate_hz": float(item), "score": 0.5})

    # Sort descending by score and limit to top_k
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k_limit]
