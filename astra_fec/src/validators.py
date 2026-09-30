"""
validators.py
Validation rules and input verification for Stage 9 FEC Candidate Testing Engine.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from .models import FECProfile


def validate_fec_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates top-level FEC configuration."""
    errors = []
    cfg = config.get("fec", config)
    if not isinstance(cfg, dict):
        return False, ["Invalid top-level FEC config format"]
        
    search_cfg = cfg.get("search", {})
    if search_cfg.get("beam_width", 8) <= 0:
        errors.append("search.beam_width must be > 0")
        
    return (len(errors) == 0), errors


def validate_interleaver_input(
    hard_bits: Optional[np.ndarray],
    soft_llrs: Optional[np.ndarray] = None
) -> Tuple[bool, str, np.ndarray, Optional[np.ndarray]]:
    """
    Validates and cleans input deinterleaved bits and soft LLRs.
    """
    if hard_bits is None or len(hard_bits) == 0:
        return False, "Input hard bitstream is empty or None", np.array([], dtype=np.uint8), None
        
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)
    if not np.all((hard_bits == 0) | (hard_bits == 1)):
        hard_bits = (hard_bits != 0).astype(np.uint8)
        
    clean_llrs = None
    if soft_llrs is not None:
        soft_llrs = np.asarray(soft_llrs, dtype=np.float32)
        if len(soft_llrs) != len(hard_bits):
            return False, f"Hard bits length ({len(hard_bits)}) != Soft LLRs length ({len(soft_llrs)})", hard_bits, None
        clean_llrs = np.nan_to_num(soft_llrs, nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)
        
    return True, "Valid input", hard_bits, clean_llrs
