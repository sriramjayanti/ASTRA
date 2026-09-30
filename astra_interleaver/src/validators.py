"""
validators.py
Validation rules and integrity checks for interleaver configs, demodulation variant inputs, and permutation maps.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from .models import InterleaverFamily, PermutationMapping


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates Stage 8 interleaver configuration dictionary.
    Rejects:
      - Negative/zero dimensions
      - Invalid branch counts / delay steps
      - Invalid orientation names
      - Negative seeds
    """
    errors = []
    cfg = config.get("interleaver", config)
    
    # Block config validation
    block_cfg = cfg.get("block", {})
    for r in block_cfg.get("row_candidates", []):
        if not isinstance(r, int) or r <= 0:
            errors.append(f"Invalid block row candidate: {r} (must be positive integer)")
    for c in block_cfg.get("column_candidates", []):
        if not isinstance(c, int) or c <= 0:
            errors.append(f"Invalid block column candidate: {c} (must be positive integer)")
    for o in block_cfg.get("orientations", []):
        if o not in ("row_to_column", "column_to_row"):
            errors.append(f"Invalid block orientation: {o}")
            
    # Convolutional config validation
    conv_cfg = cfg.get("convolutional", {})
    for b in conv_cfg.get("branch_candidates", []):
        if not isinstance(b, int) or b <= 0:
            errors.append(f"Invalid convolutional branch candidate: {b} (must be positive integer)")
    for d in conv_cfg.get("delay_steps", []):
        if not isinstance(d, int) or d <= 0:
            errors.append(f"Invalid convolutional delay step: {d} (must be positive integer)")
            
    # Helical config validation
    hel_cfg = cfg.get("helical", {})
    for r in hel_cfg.get("rows", []):
        if not isinstance(r, int) or r <= 0:
            errors.append(f"Invalid helical rows: {r} (must be positive integer)")
    for s in hel_cfg.get("steps", []):
        if not isinstance(s, int) or s <= 0:
            errors.append(f"Invalid helical step: {s} (must be positive integer)")
    for o in hel_cfg.get("orientations", []):
        if o not in ("row_diagonal", "diagonal_row"):
            errors.append(f"Invalid helical orientation: {o}")
            
    # Pseudo-random config validation
    pr_cfg = cfg.get("pseudo_random", {})
    for seed in pr_cfg.get("seed_candidates", []):
        if not isinstance(seed, int) or seed < 0:
            errors.append(f"Invalid PR seed: {seed} (must be non-negative integer)")
            
    return (len(errors) == 0), errors


def validate_demod_input(
    hard_bits: Optional[np.ndarray],
    soft_llrs: Optional[np.ndarray] = None
) -> Tuple[bool, str, np.ndarray, Optional[np.ndarray]]:
    """
    Validates and cleans input demodulation bitstream and LLRs.
    Checks:
      - Non-empty hard bits
      - Bit values in {0, 1}
      - Aligned length between hard bits and soft LLRs
      - Replaces NaNs/Infs in soft LLRs with clean bounded values
    """
    if hard_bits is None or len(hard_bits) == 0:
        return False, "Hard bits stream is empty or None", np.array([], dtype=np.uint8), None
        
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)
    
    # Check bit values
    if not np.all((hard_bits == 0) | (hard_bits == 1)):
        # Clip to {0, 1}
        hard_bits = (hard_bits != 0).astype(np.uint8)
        
    cleaned_llrs = None
    if soft_llrs is not None:
        soft_llrs = np.asarray(soft_llrs, dtype=np.float32)
        if len(soft_llrs) != len(hard_bits):
            return False, f"Hard bits length ({len(hard_bits)}) != Soft LLRs length ({len(soft_llrs)})", hard_bits, None
        cleaned_llrs = np.nan_to_num(soft_llrs, nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)
        
    return True, "Valid input", hard_bits, cleaned_llrs
