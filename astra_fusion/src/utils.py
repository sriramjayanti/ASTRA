"""
ASTRA Fusion General Utility and Formatting Functions.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger("astra_fusion")


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Configures structured logger for ASTRA Fusion Engine."""
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [ASTRA-Fusion] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def format_fusion_prediction_summary(pred: Dict[str, Any]) -> str:
    """Formats a FusionPrediction dictionary into a clean terminal report string."""
    top_k_str = ", ".join([f"{item['class']} ({item['probability']:.1%})" for item in pred.get("top_k", [])])
    evidence = pred.get("branch_evidence", {})
    ev_1d = evidence.get("resnet1d", {})
    ev_2d = evidence.get("spectrogram2d", {})

    lines = [
        "─" * 60,
        f"🎯 Predicted Class: {pred.get('predicted_class')} (Confidence: {pred.get('confidence', 0):.2%})",
        f"📊 Status:          {pred.get('status')} (Reason: {pred.get('unknown_reason', 'None')})",
        f"🤝 Agreement:       {'YES' if pred.get('branch_agreement') else 'NO'}",
        f"📈 Margin / Entropy: Margin={pred.get('confidence_margin', 0):.3f} | Entropy={pred.get('probability_entropy', 0):.3f}",
        f"🏆 Top-K:           [{top_k_str}]",
        f"🔍 1D Evidence:     {ev_1d.get('predicted_class')} ({ev_1d.get('confidence', 0):.2%})",
        f"🔍 2D Evidence:     {ev_2d.get('predicted_class')} ({ev_2d.get('confidence', 0):.2%})",
        "─" * 60,
    ]
    return "\n".join(lines)


class BenchmarkTimer:
    """Context manager and utility for precise latency profiling."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> BenchmarkTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
