"""
label_builder.py
Training-only ground-truth label generation for ASTRA Stage 11.
Evaluates candidate configurations against known ground truth parameters to assign
binary correctness labels (candidate_correct = 1 or 0) for model training and evaluation.
NEVER imported or invoked during production inference.
"""

from typing import Any, Dict, Optional


def evaluate_candidate_correctness(
    candidate: Any,
    ground_truth: Dict[str, Any],
    symbol_rate_tolerance_pct: float = 5.0,
    strict_mode: bool = True,
) -> int:
    """
    Evaluate whether candidate parameters match the true transmission parameters.
    
    Ground truth dictionary typically contains:
      - modulation (e.g. 'QPSK')
      - symbol_rate_hz (e.g. 9600.0)
      - phase_variant (e.g. 'rot0' or 'rot90')
      - interleaver_family (e.g. 'block_16x16' or 'none')
      - fec_family (e.g. 'convolutional' or 'none')
      - decoded_bits_hash (optional reference hash)
    """
    if candidate is None or ground_truth is None:
        return 0

    if hasattr(candidate, "__dict__"):
        d = getattr(candidate, "__dict__", {})
    elif isinstance(candidate, dict):
        d = candidate
    else:
        return 0

    # 1. Modulation Match
    cand_mod = str(d.get("modulation", "")).upper()
    true_mod = str(ground_truth.get("modulation", "")).upper()
    if cand_mod != true_mod:
        return 0

    # 2. Symbol Rate Match
    cand_rate = float(d.get("symbol_rate_hz", 0.0))
    true_rate = float(ground_truth.get("symbol_rate_hz", 0.0))
    if true_rate > 0.0:
        rel_diff = abs(cand_rate - true_rate) / true_rate
        if rel_diff > (symbol_rate_tolerance_pct / 100.0):
            return 0

    # 3. Interleaver Family Match
    cand_inter = str(d.get("interleaver_family", "none")).lower()
    true_inter = str(ground_truth.get("interleaver_family", "none")).lower()
    if cand_inter != true_inter:
        return 0

    # 4. FEC Family Match
    cand_fec = str(d.get("fec_family", "none")).lower()
    true_fec = str(ground_truth.get("fec_family", "none")).lower()
    if cand_fec != true_fec:
        return 0

    # 5. Phase / Decoded Bitstream Match if strict
    if strict_mode and "phase_variant" in ground_truth:
        cand_phase = str(d.get("phase_variant", "rot0")).lower()
        true_phase = str(ground_truth.get("phase_variant", "rot0")).lower()
        if cand_phase != true_phase:
            return 0

    # 6. Decoded bits hash match (equivalent pipeline check)
    if "decoded_bits_hash" in ground_truth and ground_truth["decoded_bits_hash"]:
        cand_hash = d.get("decoded_bits_hash", "")
        true_hash = ground_truth["decoded_bits_hash"]
        if cand_hash and true_hash and cand_hash == true_hash:
            return 1

    return 1
