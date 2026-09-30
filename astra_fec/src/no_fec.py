"""
no_fec.py
Pass-through / Uncoded handler (NO_FEC).
Preserves the input bitstream exactly when the received signal is uncoded.
"""

from typing import Optional, Dict, Any, Tuple
import time
import numpy as np
from .models import DecoderResult


def decode_no_fec(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray] = None
) -> DecoderResult:
    """
    Executes NO_FEC pass-through decoding.
    
    Returns:
        DecoderResult
    """
    start_time = time.perf_counter()
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)
    soft_info = np.asarray(soft_llrs, dtype=np.float32) if soft_llrs is not None else None
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    return DecoderResult(
        success=True,
        decoded_bits=hard_bits.copy(),
        decoded_soft_info=soft_info.copy() if soft_info is not None else None,
        metrics={
            "fec_family": "none",
            "code_rate": 1.0,
            "errors_corrected": 0,
            "path_metric": 0.0,
            "normalized_path_metric": 0.0,
            "syndrome_weight": 0.0,
            "parity_check_success": True
        },
        decoder_name="no_fec_passthrough",
        profile_id="none",
        runtime_ms=elapsed_ms
    )
