"""
viterbi.py
Hard-decision and soft-decision Viterbi decoder implementations.
Features traceback management, termination mode handling, path metric normalization, and LLR polarity alignment.
"""

from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np
from .models import DecoderResult, FECProfile
from .convolutional import ConvolutionalTrellis, depuncture_stream


def viterbi_decode_hard(
    hard_bits: np.ndarray,
    constraint_length: int = 7,
    generators_octal: Optional[List[int]] = None,
    termination: str = "terminated"
) -> Tuple[np.ndarray, float, float, bool]:
    """
    Fast vectorized hard-decision Viterbi decoder using Hamming distance branch metrics.
    """
    generators = generators_octal if generators_octal is not None else [171, 133]
    trellis = ConvolutionalTrellis(constraint_length, generators)
    n_out = trellis.n_outputs
    
    n_bits = len(hard_bits)
    T = n_bits // n_out
    if T == 0:
        return np.array([], dtype=np.uint8), 0.0, 0.0, False
        
    num_states = trellis.num_states
    INF_METRIC = 1e8
    
    path_metrics = np.full(num_states, INF_METRIC, dtype=np.float32)
    path_metrics[0] = 0.0
    
    traceback_state = np.zeros((T, num_states), dtype=np.int32)
    traceback_bit = np.zeros((T, num_states), dtype=np.uint8)
    
    # Precompute predecessor lookup tables for fast vectorization
    pred_s0 = np.array([trellis.prev_transitions[s][0][0] for s in range(num_states)], dtype=np.int32)
    pred_u0 = np.array([trellis.prev_transitions[s][0][1] for s in range(num_states)], dtype=np.uint8)
    pred_out0 = np.array([trellis.prev_transitions[s][0][2] for s in range(num_states)], dtype=np.uint8)
    
    pred_s1 = np.array([trellis.prev_transitions[s][1][0] for s in range(num_states)], dtype=np.int32)
    pred_u1 = np.array([trellis.prev_transitions[s][1][1] for s in range(num_states)], dtype=np.uint8)
    pred_out1 = np.array([trellis.prev_transitions[s][1][2] for s in range(num_states)], dtype=np.uint8)
    
    rx_symbols = hard_bits[:T * n_out].reshape(T, n_out)
    
    for t in range(T):
        rx = rx_symbols[t]
        
        # Branch metrics for transition 0 and 1
        bm0 = np.sum(rx != pred_out0, axis=1)
        bm1 = np.sum(rx != pred_out1, axis=1)
        
        cost0 = path_metrics[pred_s0] + bm0
        cost1 = path_metrics[pred_s1] + bm1
        
        choose1 = cost1 < cost0
        path_metrics = np.where(choose1, cost1, cost0)
        traceback_state[t] = np.where(choose1, pred_s1, pred_s0)
        traceback_bit[t] = np.where(choose1, pred_u1, pred_u0)
        
    if termination == "terminated":
        final_state = 0
        termination_match = (path_metrics[0] < INF_METRIC)
        final_metric = float(path_metrics[0])
    else:
        final_state = int(np.argmin(path_metrics))
        termination_match = True
        final_metric = float(path_metrics[final_state])
        
    decoded = np.zeros(T, dtype=np.uint8)
    curr_s = final_state
    for t in range(T - 1, -1, -1):
        decoded[t] = traceback_bit[t, curr_s]
        curr_s = traceback_state[t, curr_s]
        
    if termination == "terminated" and len(decoded) >= (constraint_length - 1):
        decoded = decoded[:-(constraint_length - 1)]
        
    norm_metric = final_metric / (len(hard_bits) + 1e-6)
    return decoded, final_metric, norm_metric, termination_match


def viterbi_decode_soft(
    soft_llrs: np.ndarray,
    constraint_length: int = 7,
    generators_octal: Optional[List[int]] = None,
    termination: str = "terminated"
) -> Tuple[np.ndarray, float, float, bool]:
    """
    Fast vectorized soft-decision Viterbi decoder using exact Log-Likelihood Ratio branch metrics.
    """
    generators = generators_octal if generators_octal is not None else [171, 133]
    trellis = ConvolutionalTrellis(constraint_length, generators)
    n_out = trellis.n_outputs
    
    n_llrs = len(soft_llrs)
    T = n_llrs // n_out
    if T == 0:
        return np.array([], dtype=np.uint8), 0.0, 0.0, False
        
    num_states = trellis.num_states
    INF_METRIC = 1e8
    
    path_metrics = np.full(num_states, INF_METRIC, dtype=np.float32)
    path_metrics[0] = 0.0
    
    traceback_state = np.zeros((T, num_states), dtype=np.int32)
    traceback_bit = np.zeros((T, num_states), dtype=np.uint8)
    
    pred_s0 = np.array([trellis.prev_transitions[s][0][0] for s in range(num_states)], dtype=np.int32)
    pred_u0 = np.array([trellis.prev_transitions[s][0][1] for s in range(num_states)], dtype=np.uint8)
    pred_out0 = np.array([trellis.prev_transitions[s][0][2] for s in range(num_states)], dtype=np.float32)
    
    pred_s1 = np.array([trellis.prev_transitions[s][1][0] for s in range(num_states)], dtype=np.int32)
    pred_u1 = np.array([trellis.prev_transitions[s][1][1] for s in range(num_states)], dtype=np.uint8)
    pred_out1 = np.array([trellis.prev_transitions[s][1][2] for s in range(num_states)], dtype=np.float32)
    
    # Expected sign mapping: 1.0 - 2.0 * out_bit
    exp_signs0 = 1.0 - 2.0 * pred_out0
    exp_signs1 = 1.0 - 2.0 * pred_out1
    
    rx_llrs = soft_llrs[:T * n_out].reshape(T, n_out).astype(np.float32)
    abs_rx_llrs = np.abs(rx_llrs)
    
    for t in range(T):
        llrs_t = rx_llrs[t]
        abs_t = abs_rx_llrs[t]
        
        bm0 = np.sum((abs_t - exp_signs0 * llrs_t) * 0.5, axis=1)
        bm1 = np.sum((abs_t - exp_signs1 * llrs_t) * 0.5, axis=1)
        
        cost0 = path_metrics[pred_s0] + bm0
        cost1 = path_metrics[pred_s1] + bm1
        
        choose1 = cost1 < cost0
        path_metrics = np.where(choose1, cost1, cost0)
        traceback_state[t] = np.where(choose1, pred_s1, pred_s0)
        traceback_bit[t] = np.where(choose1, pred_u1, pred_u0)
        
    if termination == "terminated":
        final_state = 0
        termination_match = (path_metrics[0] < INF_METRIC)
        final_metric = float(path_metrics[0])
    else:
        final_state = int(np.argmin(path_metrics))
        termination_match = True
        final_metric = float(path_metrics[final_state])
        
    decoded = np.zeros(T, dtype=np.uint8)
    curr_s = final_state
    for t in range(T - 1, -1, -1):
        decoded[t] = traceback_bit[t, curr_s]
        curr_s = traceback_state[t, curr_s]
        
    if termination == "terminated" and len(decoded) >= (constraint_length - 1):
        decoded = decoded[:-(constraint_length - 1)]
        
    norm_metric = final_metric / (len(soft_llrs) + 1e-6)
    return decoded, final_metric, norm_metric, termination_match


def decode_convolutional_profile(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    profile: FECProfile,
    prefer_soft: bool = True
) -> DecoderResult:
    """
    Executes Viterbi decoding for a given convolutional FEC profile with depuncturing and telemetry.
    """
    start_time = time.perf_counter()
    K = profile.constraint_length or 7
    gens = profile.generators_octal or [171, 133]
    term = profile.termination or "terminated"
    punct_pat = profile.puncturing_pattern
    
    is_punctured = (punct_pat is not None and len(punct_pat) > 0)
    
    if is_punctured:
        # If soft LLRs not provided, create +/- 5.0 soft LLRs from hard bits
        # so punctured positions receive true neutral (0.0) branch metric without hard 0 bias
        base_soft = soft_llrs if soft_llrs is not None else np.where(hard_bits == 0, 5.0, -5.0).astype(np.float32)
        clean_hard, clean_soft = depuncture_stream(hard_bits, base_soft, punct_pat)
        use_soft = True
    else:
        clean_hard = hard_bits
        clean_soft = soft_llrs
        use_soft = prefer_soft and (clean_soft is not None) and len(clean_soft) > 0
    
    if use_soft and clean_soft is not None:
        # Soft Viterbi
        decoded_bits, raw_metric, norm_metric, term_match = viterbi_decode_soft(
            soft_llrs=clean_soft,
            constraint_length=K,
            generators_octal=gens,
            termination=term
        )
        decoder_name = "soft_viterbi"
    else:
        # Hard Viterbi fallback
        decoded_bits, raw_metric, norm_metric, term_match = viterbi_decode_hard(
            hard_bits=clean_hard,
            constraint_length=K,
            generators_octal=gens,
            termination=term
        )
        decoder_name = "hard_viterbi"
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    success = (len(decoded_bits) > 0) and term_match
    
    # Estimated errors corrected (rough estimate based on path metric)
    est_errors = int(round(raw_metric)) if not use_soft else int(round(raw_metric / 6.0))
    
    return DecoderResult(
        success=success,
        decoded_bits=decoded_bits,
        decoded_soft_info=None,
        metrics={
            "fec_family": "convolutional",
            "profile_id": profile.profile_id,
            "decoder_name": decoder_name,
            "code_rate": profile.rate,
            "path_metric": float(raw_metric),
            "normalized_path_metric": float(norm_metric),
            "termination_match": bool(term_match),
            "estimated_errors_corrected": est_errors,
            "is_punctured": (punct_pat is not None and len(punct_pat) > 0),
            "used_soft": use_soft
        },
        failure_reason=None if success else "Viterbi termination mismatch or empty stream",
        decoder_name=decoder_name,
        profile_id=profile.profile_id,
        runtime_ms=elapsed_ms
    )
