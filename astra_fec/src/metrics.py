"""
metrics.py
Standardized metric normalization and feature extraction across heterogeneous FEC decoder families.
"""

from typing import Dict, Any, Tuple
import numpy as np
from .models import DecoderResult, FECFamily


def normalize_decoder_metrics(decoder_result: DecoderResult, family: str) -> Dict[str, Any]:
    """
    Standardizes raw metrics from heterogeneous decoders into unified telemetry features.
    """
    metrics = decoder_result.metrics or {}
    
    # Defaults
    normalized_path_metric = 0.0
    syndrome_score = 0.0
    parity_success = False
    converged = False
    errors_corrected = 0
    code_rate = float(metrics.get("code_rate", 1.0))
    
    if family in ("convolutional", "conv"):
        raw_pm = float(metrics.get("path_metric", 0.0))
        norm_pm = float(metrics.get("normalized_path_metric", 0.0))
        term_match = bool(metrics.get("termination_match", True))
        errors_corrected = int(metrics.get("estimated_errors_corrected", 0))
        
        normalized_path_metric = norm_pm
        parity_success = term_match and (norm_pm < 0.35)
        # Low path metric implies high syndrome/consistency score
        syndrome_score = float(np.clip(1.0 - norm_pm * 2.5, 0.0, 1.0)) if term_match else 0.0
        converged = decoder_result.success
        
    elif family in ("reed_solomon", "rs"):
        uncorrectable = int(metrics.get("uncorrectable_codewords", 0))
        corrected_syms = int(metrics.get("corrected_symbols", 0))
        total_cw = int(metrics.get("total_codewords", 1))
        
        errors_corrected = corrected_syms
        parity_success = (uncorrectable == 0) and (total_cw > 0)
        syndrome_score = 1.0 if parity_success else 0.0
        converged = parity_success
        normalized_path_metric = float(uncorrectable) / float(max(1, total_cw))
        
    elif family in ("ldpc",):
        conv = bool(metrics.get("converged", False))
        init_wt = float(metrics.get("initial_syndrome_weight", 0.0))
        final_wt = float(metrics.get("final_syndrome_weight", 0.0))
        iters = int(metrics.get("iterations_used", 0))
        
        converged = conv
        parity_success = conv
        syndrome_score = 1.0 if conv else float(np.clip(1.0 - final_wt / max(1.0, init_wt), 0.0, 1.0))
        normalized_path_metric = final_wt / max(1.0, init_wt + 1.0)
        
    elif family in ("concatenated",):
        inner_pm = float(metrics.get("inner_viterbi_metric", 0.0))
        outer_success = bool(metrics.get("outer_rs_success", False))
        outer_corr = int(metrics.get("outer_rs_corrected_symbols", 0))
        
        errors_corrected = outer_corr
        parity_success = outer_success
        converged = outer_success
        syndrome_score = 1.0 if outer_success else 0.0
        normalized_path_metric = inner_pm
        
    elif family in ("none", "identity"):
        parity_success = True
        syndrome_score = 0.5  # Neutral baseline
        converged = True
        normalized_path_metric = 0.0
        
    return {
        "normalized_path_metric": float(normalized_path_metric),
        "syndrome_score": float(syndrome_score),
        "parity_success": bool(parity_success),
        "converged": bool(converged),
        "errors_corrected": int(errors_corrected),
        "code_rate": float(code_rate),
        "decoder_success": bool(decoder_result.success)
    }
