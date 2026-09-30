"""
utils.py
Synthetic interleaver data generator, round-trip verification, recall evaluation, and explainability reporting.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from .models import InterleaverFamily, InterleaverTestResult, InterleaverCandidateResult
from .block import interleave_block
from .convolutional import interleave_convolutional
from .helical import interleave_helical
from .pseudo_random import interleave_pseudorandom
from .identity import interleave_identity


def generate_synthetic_interleaved_stream(
    bit_count: int = 1024,
    pattern_type: str = "pn9",
    interleaver_family: str = "block",
    interleaver_params: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates a synthetic bitstream with known ground-truth interleaver.
    
    Returns:
        (raw_hard_bits, raw_soft_llrs, interleaved_hard_bits, interleaved_soft_llrs, ground_truth_meta)
    """
    rng = np.random.default_rng(seed)
    
    if pattern_type == "pn9":
        # Generate pseudo-noise binary sequence
        raw_bits = rng.integers(0, 2, size=bit_count, dtype=np.uint8)
    elif pattern_type == "structured_frame":
        # Repeated 32-bit telemetry frame sync word (0xEB907E5A) + periodic frame structure
        sync = np.array([
            1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0
        ], dtype=np.uint8)
        frame_len = 64
        num_frames = bit_count // frame_len
        frames = []
        for _ in range(num_frames):
            payload = rng.integers(0, 2, size=frame_len - len(sync), dtype=np.uint8)
            frames.append(np.concatenate([sync, payload]))
        raw_bits = np.concatenate(frames)
    else:
        raw_bits = rng.integers(0, 2, size=bit_count, dtype=np.uint8)
        
    # Generate aligned synthetic LLRs: +6.0 for bit 0, -6.0 for bit 1 + small noise
    noise = rng.normal(0.0, 0.5, size=len(raw_bits)).astype(np.float32)
    raw_llrs = (np.where(raw_bits == 0, 6.0, -6.0) + noise).astype(np.float32)
    
    params = interleaver_params or {}
    
    if interleaver_family in ("identity", "none"):
        int_hard, int_soft = interleave_identity(raw_bits, raw_llrs)
    elif interleaver_family == "block":
        r = params.get("rows", 16)
        c = params.get("cols", 16)
        orient = params.get("orientation", "row_to_column")
        int_hard, int_soft = interleave_block(raw_bits, raw_llrs, rows=r, cols=c, orientation=orient)
    elif interleaver_family == "convolutional":
        b = params.get("branches", 4)
        d = params.get("delay_step", 2)
        int_hard, int_soft = interleave_convolutional(raw_bits, raw_llrs, branch_count=b, delay_step=d)
    elif interleaver_family == "helical":
        r = params.get("rows", 16)
        c = params.get("cols", 16)
        step = params.get("step", 1)
        orient = params.get("orientation", "row_diagonal")
        int_hard, int_soft = interleave_helical(raw_bits, raw_llrs, rows=r, cols=c, step=step, orientation=orient)
    elif interleaver_family == "pseudo_random":
        length = params.get("length", 256)
        s = params.get("seed", 42)
        algo = params.get("algorithm", "pcg64")
        int_hard, int_soft = interleave_pseudorandom(raw_bits, raw_llrs, length=length, seed=s, algorithm=algo)
    else:
        raise ValueError(f"Unknown synthetic interleaver family: {interleaver_family}")
        
    meta = {
        "family": interleaver_family,
        "parameters": params,
        "bit_count": len(raw_bits),
        "pattern_type": pattern_type
    }
    return raw_bits, raw_llrs, int_hard, int_soft, meta


def evaluate_candidate_recall(
    test_result: InterleaverTestResult,
    ground_truth_family: str,
    ground_truth_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates whether the true deinterleaver was generated and retained in Top-K.
    """
    survivors = test_result.surviving_candidates
    
    def matches_ground_truth(candidate: InterleaverCandidateResult) -> bool:
        if candidate.interleaver_family != ground_truth_family:
            return False
        for k, v in ground_truth_params.items():
            if candidate.parameters.get(k) != v:
                return False
        return True

    found_in_survivors = False
    rank_in_survivors = -1
    for rank, cand in enumerate(survivors):
        if matches_ground_truth(cand):
            found_in_survivors = True
            rank_in_survivors = rank + 1
            break
            
    return {
        "ground_truth_family": ground_truth_family,
        "ground_truth_params": ground_truth_params,
        "surviving_count": len(survivors),
        "found_in_top_k": found_in_survivors,
        "rank": rank_in_survivors,
        "top_1_family": test_result.top_candidate.interleaver_family if test_result.top_candidate else None
    }


def format_interleaver_summary(test_result: InterleaverTestResult) -> str:
    """Formats human-readable summary report of tested interleaver hypotheses."""
    lines = [
        "==================================================",
        f"ASTRA STAGE 8: INTERLEAVER CANDIDATE TESTING REPORT",
        "==================================================",
        f"Demodulation Variant: {test_result.demod_variant_id}",
        f"Tested Candidates:    {test_result.tested_candidates_count}",
        f"Surviving Beam:       {len(test_result.surviving_candidates)}",
        f"Execution Time:       {test_result.execution_time_ms:.2f} ms",
        "--------------------------------------------------",
        "TOP SURVIVING HYPOTHESES:"
    ]
    for idx, c in enumerate(test_result.surviving_candidates[:5]):
        lines.append(
            f"  #{idx+1}: [{c.interleaver_family.upper()}] ID={c.interleaver_candidate_id} | "
            f"Score={c.overall_interleaver_score:.4f} | Status={c.status.value} | "
            f"Params={c.parameters}"
        )
    lines.append("==================================================")
    return "\n".join(lines)
