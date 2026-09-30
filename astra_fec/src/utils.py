"""
utils.py
Synthetic FEC stream generation, BER computation, candidate recall evaluation, and explainability reporting.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from .models import FECTestResult, FECCandidateResult, FECProfile
from .profiles import get_profile_registry
from .convolutional import encode_convolutional
from .reed_solomon import ReedSolomonCodec, bits_to_symbols, symbols_to_bits
from .ldpc import construct_qc_ldpc_matrix, LDPCCodec


def generate_synthetic_fec_stream(
    msg_len: int = 256,
    fec_family: str = "convolutional",
    profile_id: str = "conv_k7_r12_nasa",
    snr_db: float = 10.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates synthetic message bits, encodes with known ground-truth FEC profile, and adds AWGN/channel noise.
    
    Returns:
        (raw_message_bits, channel_hard_bits, channel_soft_llrs, metadata)
    """
    rng = np.random.default_rng(seed)
    registry = get_profile_registry()
    
    prof = registry.get_profile(profile_id)
    
    if fec_family in ("none", "uncoded"):
        raw_msg = rng.integers(0, 2, size=msg_len, dtype=np.uint8)
        encoded_bits = raw_msg.copy()
    elif fec_family == "convolutional":
        K = prof.constraint_length if prof else 7
        gens = prof.generators_octal if prof else [171, 133]
        term = prof.termination if prof else "terminated"
        punct = prof.puncturing_pattern if prof else None
        
        raw_msg = rng.integers(0, 2, size=msg_len, dtype=np.uint8)
        encoded_bits = encode_convolutional(
            message_bits=raw_msg,
            constraint_length=K,
            generators_octal=gens,
            termination=term,
            puncturing_pattern=punct
        )
    elif fec_family == "reed_solomon":
        m = prof.symbol_bits if prof else 8
        n = prof.n if prof else 255
        k = prof.k if prof else 223
        fcr = prof.fcr if prof else 0
        prim = prof.prim_poly if prof else 0x11D
        
        short_n = prof.shortened_n if prof else None
        short_k = prof.shortened_k if prof else None
        active_k = short_k if short_k else k
        active_n = short_n if short_n else n
        
        # Exact message symbols
        msg_syms = rng.integers(0, 1 << m, size=active_k).tolist()
        raw_msg = symbols_to_bits(msg_syms, m=m)
        
        codec = ReedSolomonCodec(n=n, k=k, m=m, prim_poly=prim, fcr=fcr)
        if short_n:
            pad_len = n - active_n
            cw_full = codec.encode([0] * pad_len + msg_syms)
            cw_short = cw_full[pad_len:]
            encoded_bits = symbols_to_bits(cw_short, m=m)
        else:
            cw = codec.encode(msg_syms)
            encoded_bits = symbols_to_bits(cw, m=m)
    elif fec_family == "ldpc":
        n = prof.n if prof else 128
        k = prof.k if prof else 64
        block_sz = prof.block_size if prof else 16
        
        H = construct_qc_ldpc_matrix(n=n, k=k, block_size=block_sz)
        codec = LDPCCodec(H)
        
        raw_msg = rng.integers(0, 2, size=k, dtype=np.uint8)
        encoded_bits = codec.encode(raw_msg)
    else:
        raw_msg = rng.integers(0, 2, size=msg_len, dtype=np.uint8)
        encoded_bits = raw_msg.copy()
        
    # Add Gaussian noise
    sigma = 10.0 ** (-snr_db / 20.0)
    # BPSK mapping: bit 0 -> +1.0, bit 1 -> -1.0
    mod_bpsk = 1.0 - 2.0 * encoded_bits.astype(np.float32)
    noise = rng.normal(0.0, sigma, size=len(encoded_bits)).astype(np.float32)
    rx_signal = mod_bpsk + noise
    
    # Compute true LLR: 2 * y / sigma^2
    llrs = (2.0 * rx_signal / (sigma ** 2 + 1e-6)).astype(np.float32)
    # Hard slice: LLR >= 0 -> bit 0, LLR < 0 -> bit 1
    hard_bits = (llrs < 0).astype(np.uint8)
    
    meta = {
        "fec_family": fec_family,
        "profile_id": profile_id,
        "msg_len": len(raw_msg),
        "encoded_len": len(encoded_bits),
        "snr_db": snr_db
    }
    return raw_msg, hard_bits, llrs, meta


def compute_ber(recovered_bits: np.ndarray, ground_truth_bits: np.ndarray) -> float:
    """Computes Bit Error Rate between recovered bits and ground-truth bits."""
    min_len = min(len(recovered_bits), len(ground_truth_bits))
    if min_len == 0:
        return 1.0
    errors = np.sum(recovered_bits[:min_len] != ground_truth_bits[:min_len])
    return float(errors) / float(min_len)


def evaluate_fec_candidate_recall(
    test_result: FECTestResult,
    ground_truth_profile_id: Optional[str]
) -> Dict[str, Any]:
    """
    Evaluates whether the true FEC profile was generated and retained in the Top-K beam.
    """
    survivors = test_result.surviving_candidates
    found = False
    rank = -1
    
    for r, c in enumerate(survivors):
        cand_prof = c.fec_parameters.get("profile_id")
        if ground_truth_profile_id in (None, "none") and c.fec_family == "none":
            found = True
            rank = r + 1
            break
        elif cand_prof == ground_truth_profile_id:
            found = True
            rank = r + 1
            break
            
    return {
        "ground_truth_profile_id": ground_truth_profile_id,
        "found_in_top_k": found,
        "rank": rank,
        "surviving_count": len(survivors),
        "top_1_profile": test_result.top_candidate.fec_parameters.get("profile_id") if test_result.top_candidate else None
    }


def format_fec_summary(test_result: FECTestResult) -> str:
    """Formats human-readable summary of tested FEC hypotheses."""
    lines = [
        "==================================================",
        f"ASTRA STAGE 9: FEC CANDIDATE TESTING REPORT",
        "==================================================",
        f"Lineage ID:        {test_result.interleaver_candidate_id}",
        f"Demod Variant:     {test_result.demod_variant_id}",
        f"Tested Hypotheses: {test_result.tested_candidates_count}",
        f"Surviving Beam:    {len(test_result.surviving_candidates)}",
        f"Execution Time:    {test_result.execution_time_ms:.2f} ms",
        "--------------------------------------------------",
        "TOP SURVIVING FEC CANDIDATES:"
    ]
    for idx, c in enumerate(test_result.surviving_candidates[:5]):
        prof_name = c.fec_parameters.get("profile_id", c.fec_family)
        lines.append(
            f"  #{idx+1}: [{c.fec_family.upper()}] Profile={prof_name} | "
            f"Score={c.fec_quality_score:.4f} | Status={c.decoder_status.value} | "
            f"Rate={c.code_rate:.2f} | OutBits={c.output_bit_count}"
        )
    lines.append("==================================================")
    return "\n".join(lines)
