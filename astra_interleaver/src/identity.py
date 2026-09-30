"""
identity.py
Identity / Pass-through deinterleaver (NO_INTERLEAVER).
Guarantees output bits == input bits and output LLRs == input LLRs.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from .models import PermutationMapping, InterleaverFamily
from .permutation import compute_permutation_hash


def create_identity_mapping(length: int) -> PermutationMapping:
    """
    Generates an identity permutation mapping of size length.
    forward_indices = [0, 1, ..., length-1]
    inverse_indices = [0, 1, ..., length-1]
    """
    indices = np.arange(length, dtype=np.int64)
    return PermutationMapping(
        forward_indices=indices.copy(),
        inverse_indices=indices.copy(),
        length=length,
        family=InterleaverFamily.IDENTITY.value,
        parameters={"length": length, "type": "identity"},
        mapping_hash=compute_permutation_hash(indices)
    )


def deinterleave_identity(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], PermutationMapping]:
    """
    Applies identity deinterleaving (no-op pass-through).
    Preserves exact bit values and soft LLRs.
    """
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)
    soft_llrs = np.asarray(soft_llrs, dtype=np.float32) if soft_llrs is not None else None
    length = len(hard_bits)
    mapping = create_identity_mapping(length)
    return hard_bits.copy(), soft_llrs.copy() if soft_llrs is not None else None, mapping


def interleave_identity(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Identity interleaver for synthetic generation."""
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)
    soft_llrs = np.asarray(soft_llrs, dtype=np.float32) if soft_llrs is not None else None
    return hard_bits.copy(), soft_llrs.copy() if soft_llrs is not None else None
