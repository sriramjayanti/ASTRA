"""
utils.py
Utilities and synthetic helper generators for ASTRA Candidate Engine testing.
"""

from typing import Dict, Any, List


def create_mock_fusion_prediction(
    top_k: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a realistic mock Fusion Engine output."""
    if top_k is None:
        top_k = [
            {"class": "QPSK", "probability": 0.80},
            {"class": "8PSK", "probability": 0.12},
            {"class": "16-QAM", "probability": 0.05}
        ]
    probs = {item["class"]: item["probability"] for item in top_k}
    return {
        "top_k": top_k,
        "probabilities": probs,
        "confidence": top_k[0]["probability"] if top_k else 0.0,
        "status": "ESTIMATED"
    }


def create_mock_symbol_rate_prediction(
    top_k: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a realistic mock Symbol-Rate Estimation Engine output."""
    if top_k is None:
        top_k = [
            {"symbol_rate_hz": 9600.0, "score": 0.85, "samples_per_symbol": 20.0},
            {"symbol_rate_hz": 4800.0, "score": 0.10, "samples_per_symbol": 40.0},
            {"symbol_rate_hz": 19200.0, "score": 0.05, "samples_per_symbol": 10.0}
        ]
    return {
        "top_k": top_k,
        "estimated_baud": top_k[0]["symbol_rate_hz"] if top_k else 0.0,
        "confidence": top_k[0]["score"] if top_k else 0.0,
        "status": "ESTIMATED"
    }


def create_mock_rf_support(
    family_probs: Dict[str, float] = None
) -> Dict[str, Any]:
    """Create a realistic mock Random Forest Support Model output."""
    if family_probs is None:
        family_probs = {
            "PSK": 0.92,
            "FSK": 0.04,
            "QAM": 0.03,
            "ANALOG": 0.01,
            "UNKNOWN": 0.00
        }
    return {
        "predicted_family": "PSK",
        "family_probabilities": family_probs,
        "quality": {"snr_class": "HIGH_SNR", "corruption": "CLEAN"}
    }


def create_mock_constellation_evidence(
    modulation_support: Dict[str, float] = None,
    stage: str = "pre_sync"
) -> Dict[str, Any]:
    """Create a realistic mock Constellation Analysis output."""
    if modulation_support is None:
        modulation_support = {
            "QPSK": 0.88,
            "8PSK": 0.35,
            "16-QAM": 0.12,
            "BPSK": 0.20
        }
    return {
        "stage": stage,
        "modulation_support": modulation_support,
        "estimated_clusters": 4
    }
