"""
ASTRA Headless Signal Analyzer (CLI).
Provides end-to-end signal processing and recovery without requiring GUI.
Reads raw IQ captures, executes pipeline stages, and outputs structured JSON or text reports.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from astra_config.loader import load_astra_config
from astra_config.schema import ConfigurationError


def load_iq_file(file_path: Path) -> np.ndarray:
    """
    Safely loads IQ samples from disk.
    Supports .npy (complex64 or [2, N]), .iq/.bin (raw interleaved float32).
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"IQ file not found: {file_path}")

    size_bytes = file_path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"File is empty: {file_path}")
    if size_bytes > 500 * 1024 * 1024:
        raise ValueError(f"File exceeds maximum allowed size (500 MB): {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(str(file_path))
        if np.iscomplexobj(data):
            return data.astype(np.complex64).flatten()
        elif data.ndim == 2 and data.shape[0] == 2:
            return (data[0] + 1j * data[1]).astype(np.complex64)
        elif data.ndim == 1 and len(data) % 2 == 0:
            return (data[0::2] + 1j * data[1::2]).astype(np.complex64)
        else:
            raise ValueError(f"Unsupported numpy IQ array shape: {data.shape}")
    else:
        # Assume interleaved 32-bit float IQ
        raw = np.fromfile(str(file_path), dtype=np.float32)
        if len(raw) < 2:
            raise ValueError(f"Insufficient samples in file: {file_path}")
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def analyze_signal(
    file_path: str,
    sample_rate: float = 192000.0,
    config_dir: Optional[str] = None,
    output_format: str = "text"
) -> Dict[str, Any]:
    """
    Executes the end-to-end ASTRA scientific pipeline on a signal capture.
    """
    t0 = time.monotonic()
    path = Path(file_path)
    iq_data = load_iq_file(path)

    n_samples = len(iq_data)
    mean_pwr = float(np.mean(np.abs(iq_data) ** 2))
    pwr_dbfs = 10.0 * np.log10(mean_pwr + 1e-12)

    # 1. DSP Preprocessing (DC removal, RMS normalize)
    iq_centered = iq_data - np.mean(iq_data)
    rms = np.sqrt(np.mean(np.abs(iq_centered) ** 2)) + 1e-12
    iq_norm = (iq_centered / rms).astype(np.complex64)

    results: Dict[str, Any] = {
        "source_file": str(path.name),
        "sample_count": n_samples,
        "sample_rate_hz": sample_rate,
        "duration_ms": round((n_samples / sample_rate) * 1000.0, 2),
        "power_dbfs": round(pwr_dbfs, 2),
        "stages": {}
    }

    # 2. Modulation Classification & Fusion
    pred_mod = "UNKNOWN"
    mod_conf = 0.0
    try:
        from astra_fusion.src.inference import ASTRAFusionEngine
        engine = ASTRAFusionEngine(device="cpu")
        fusion_pred = engine.predict(iq_norm[:2048])
        pred_mod = fusion_pred.predicted_class
        mod_conf = float(fusion_pred.confidence)
        results["stages"]["modulation"] = {
            "predicted_class": pred_mod,
            "confidence": round(mod_conf, 4),
            "top_candidates": [
                {"class": getattr(c, "class_name", str(c)), "prob": round(float(getattr(c, "probability", 0.0)), 4)}
                for c in fusion_pred.top_k[:3]
            ]
        }
    except Exception as e:
        results["stages"]["modulation"] = {
            "predicted_class": "UNKNOWN",
            "confidence": 0.0,
            "error": str(e)
        }

    # 3. Symbol Rate Estimation
    pred_baud = 0.0
    sr_conf = 0.0
    try:
        from astra_symbol_rate.src.inference import SymbolRateInferenceEngine
        sr_engine = SymbolRateInferenceEngine(model_path="checkpoints/symbol_rate_ranker.joblib")
        sr_res = sr_engine.predict(iq_norm, sample_rate=sample_rate)
        pred_baud = float(sr_res.estimated_symbol_rate)
        sr_conf = float(sr_res.confidence)
        results["stages"]["symbol_rate"] = {
            "estimated_baud": round(pred_baud, 1),
            "sps": round(sample_rate / pred_baud, 2) if pred_baud > 0 else 0.0,
            "confidence": round(sr_conf, 4)
        }
    except Exception as e:
        results["stages"]["symbol_rate"] = {
            "estimated_baud": 0.0,
            "confidence": 0.0,
            "error": str(e)
        }

    # 4. Constellation Support AI
    try:
        from astra_random_forest.src.inference import ConstellationRFInferenceEngine
        rf_engine = ConstellationRFInferenceEngine(checkpoint_path="checkpoints/random_forest_support.joblib")
        rf_res = rf_engine.predict_from_iq(iq_norm[:4096])
        results["stages"]["constellation"] = {
            "family": rf_res.predicted_family,
            "confidence": round(float(rf_res.confidence), 4)
        }
    except Exception as e:
        results["stages"]["constellation"] = {"family": "UNKNOWN", "confidence": 0.0, "error": str(e)}

    # 5. Explainability Breakdown
    try:
        from astra_explainability.src.inference import ExplainabilityEngine
        from astra_explainability.src.schema import (
            SignalRecord, StageResult, FieldExplanation, FieldStatus, OverallStatus
        )
        exp_engine = ExplainabilityEngine()

        overall_conf = (mod_conf * 0.4 + sr_conf * 0.3) if (mod_conf > 0 and sr_conf > 0) else 0.0
        status = "CONFIRMED" if overall_conf > 0.85 else ("ESTIMATED" if overall_conf > 0.50 else "UNKNOWN")

        results["summary"] = {
            "overall_status": status,
            "overall_confidence": round(overall_conf, 4),
            "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2)
        }
    except Exception as e:
        results["summary"] = {
            "overall_status": "UNKNOWN",
            "overall_confidence": 0.0,
            "error": str(e)
        }

    return results
