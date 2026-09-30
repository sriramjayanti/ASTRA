"""
pseudo_random.py
Pseudo-random permutation interleaver and deinterleaver family.
Enforces bounded, constrained search spaces (known seed lists, profiles) to prevent combinatorial explosion.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from .models import PermutationMapping, InterleaverFamily
from .permutation import (
    invert_permutation,
    validate_permutation,
    compute_permutation_hash,
    apply_block_chunked_inverse
)


def generate_pseudorandom_indices(length: int, seed: int, algorithm: str = "pcg64") -> np.ndarray:
    """
    Generates a deterministic pseudorandom permutation of length N using specified algorithm and seed.
    """
    if length <= 0:
        raise ValueError(f"Permutation length ({length}) must be positive")
    if seed < 0:
        raise ValueError(f"Seed ({seed}) must be non-negative")
        
    if algorithm in ("pcg64", "numpy_default"):
        # NumPy default_rng uses PCG64 bit generator
        rng = np.random.default_rng(seed)
        forward = rng.permutation(length).astype(np.int64)
    elif algorithm == "fisher_yates":
        # Classical Fisher-Yates with Mersenne Twister RandomState
        rng = np.random.RandomState(seed)
        forward = rng.permutation(length).astype(np.int64)
    else:
        raise ValueError(f"Unsupported pseudorandom algorithm: {algorithm}. Supported: 'pcg64', 'fisher_yates'")
        
    return forward


def create_pseudorandom_mapping(
    length: int,
    seed: int,
    algorithm: str = "pcg64",
    profile_name: Optional[str] = None
) -> PermutationMapping:
    """
    Creates a PermutationMapping for a pseudo-random permutation.
    """
    forward_indices = generate_pseudorandom_indices(length, seed, algorithm)
    inverse_indices = invert_permutation(forward_indices)
    
    is_valid, reason = validate_permutation(inverse_indices, expected_length=length)
    if not is_valid:
        raise ValueError(f"Invalid pseudorandom permutation: {reason}")
        
    params = {
        "seed": int(seed),
        "length": int(length),
        "algorithm": algorithm
    }
    if profile_name:
        params["profile_name"] = profile_name
        
    return PermutationMapping(
        forward_indices=forward_indices,
        inverse_indices=inverse_indices,
        length=length,
        family=InterleaverFamily.PSEUDO_RANDOM.value,
        parameters=params,
        mapping_hash=compute_permutation_hash(inverse_indices)
    )


def interleave_pseudorandom(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    length: int,
    seed: int,
    algorithm: str = "pcg64"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Interleaves bitstream using pseudo-random permutation.
    """
    mapping = create_pseudorandom_mapping(length, seed, algorithm)
    B = mapping.length
    N = len(hard_bits)
    num_blocks = N // B
    if num_blocks == 0:
        raise ValueError(f"Bitstream length {N} is less than PR block size {B}")
        
    usable_len = num_blocks * B
    hard_usable = hard_bits[:usable_len].reshape(num_blocks, B)
    interleaved_hard = hard_usable[:, mapping.forward_indices].ravel()
    
    interleaved_soft = None
    if soft_llrs is not None:
        soft_usable = soft_llrs[:usable_len].reshape(num_blocks, B)
        interleaved_soft = soft_usable[:, mapping.forward_indices].ravel()
        
    return interleaved_hard, interleaved_soft


def deinterleave_pseudorandom(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    length: int,
    seed: int,
    algorithm: str = "pcg64",
    padding_policy: str = "truncate_tail"
) -> Tuple[np.ndarray, Optional[np.ndarray], PermutationMapping, int]:
    """
    Deinterleaves bitstream using pseudo-random permutation.
    """
    mapping = create_pseudorandom_mapping(length, seed, algorithm)
    deint_hard, deint_soft, remainder = apply_block_chunked_inverse(
        hard_bits=hard_bits,
        soft_llrs=soft_llrs,
        block_inverse_indices=mapping.inverse_indices,
        padding_policy=padding_policy
    )
    return deint_hard, deint_soft, mapping, remainder
