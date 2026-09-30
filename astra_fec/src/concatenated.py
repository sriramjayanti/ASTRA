"""
concatenated.py
Concatenated FEC decoder engine.
Executes chained decoding in exact reverse order of encoding (Inner Viterbi -> Outer Reed-Solomon).
"""

from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np
from .models import DecoderResult, FECProfile
from .profiles import get_profile_registry
from .viterbi import decode_convolutional_profile
from .reed_solomon import decode_reed_solomon_profile


def decode_concatenated_profile(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    profile: FECProfile,
    prefer_soft: bool = True
) -> DecoderResult:
    """
    Executes concatenated decoding:
      1. Inner Convolutional/Viterbi decode on received channel bits & soft LLRs
      2. Outer Reed-Solomon decode on intermediate deinterleaved/decoded bits
    """
    start_time = time.perf_counter()
    registry = get_profile_registry()
    
    outer_pid = profile.outer_profile
    inner_pid = profile.inner_profile
    
    outer_prof = registry.get_profile(outer_pid) if outer_pid else None
    inner_prof = registry.get_profile(inner_pid) if inner_pid else None
    
    if inner_prof is None or outer_prof is None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DecoderResult(
            success=False,
            decoded_bits=np.array([], dtype=np.uint8),
            failure_reason=f"Missing inner ({inner_pid}) or outer ({outer_pid}) profile in registry",
            decoder_name="concatenated",
            profile_id=profile.profile_id,
            runtime_ms=elapsed_ms
        )
        
    # Stage 1: Inner Viterbi Decoding
    inner_res = decode_convolutional_profile(
        hard_bits=hard_bits,
        soft_llrs=soft_llrs,
        profile=inner_prof,
        prefer_soft=prefer_soft
    )
    
    if not inner_res.success or len(inner_res.decoded_bits) == 0:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DecoderResult(
            success=False,
            decoded_bits=np.array([], dtype=np.uint8),
            metrics={
                "fec_family": "concatenated",
                "inner_stage_failed": True,
                "inner_metrics": inner_res.metrics
            },
            failure_reason="Inner Viterbi decoding failed",
            decoder_name="concatenated",
            profile_id=profile.profile_id,
            runtime_ms=elapsed_ms
        )
        
    # Stage 2: Outer Reed-Solomon Decoding
    outer_res = decode_reed_solomon_profile(
        hard_bits=inner_res.decoded_bits,
        profile=outer_prof
    )
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    overall_success = outer_res.success
    
    combined_rate = (inner_prof.rate) * (outer_res.metrics.get("code_rate", 1.0))
    
    return DecoderResult(
        success=overall_success,
        decoded_bits=outer_res.decoded_bits,
        decoded_soft_info=None,
        metrics={
            "fec_family": "concatenated",
            "profile_id": profile.profile_id,
            "overall_code_rate": round(combined_rate, 4),
            "inner_profile": inner_pid,
            "outer_profile": outer_pid,
            "inner_viterbi_metric": inner_res.metrics.get("normalized_path_metric", 0.0),
            "outer_rs_corrected_symbols": outer_res.metrics.get("corrected_symbols", 0),
            "outer_rs_uncorrectable": outer_res.metrics.get("uncorrectable_codewords", 0),
            "outer_rs_success": outer_res.success
        },
        failure_reason=outer_res.failure_reason,
        decoder_name="concatenated_rs_viterbi",
        profile_id=profile.profile_id,
        runtime_ms=elapsed_ms
    )
