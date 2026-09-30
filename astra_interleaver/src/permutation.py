"""
permutation.py
Core permutation mathematics, inversion algorithms, hash generation, and aligned hard/soft array reordering.
"""

from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import hashlib


def invert_permutation(forward_indices: np.ndarray) -> np.ndarray:
    """
    Given forward gather indices p where y = x[p],
    computes inverse gather indices inv such that x = y[inv].
    Mathematically: inv[p[i]] = i.
    """
    forward_indices = np.asarray(forward_indices, dtype=np.int64)
    n = len(forward_indices)
    inverse = np.zeros(n, dtype=np.int64)
    inverse[forward_indices] = np.arange(n, dtype=np.int64)
    return inverse


def validate_permutation(indices: np.ndarray, expected_length: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validates that an integer array forms a true bijective permutation of [0, N-1].
    Checks:
      1. Correct length
      2. No duplicates
      3. No missing indices
      4. Correct bounds [0, N-1]
    """
    if indices is None:
        return False, "Indices array is None"
    indices = np.asarray(indices)
    if indices.ndim != 1:
        return False, f"Indices array must be 1D, got shape {indices.shape}"
    n = len(indices)
    if n == 0:
        return False, "Indices array is empty"
    if expected_length is not None and n != expected_length:
        return False, f"Length {n} does not match expected length {expected_length}"
    
    # Check bounds
    if np.min(indices) < 0 or np.max(indices) >= n:
        return False, f"Index out of range [0, {n-1}] (min={np.min(indices)}, max={np.max(indices)})"
    
    # Check uniqueness
    unique_count = len(np.unique(indices))
    if unique_count != n:
        return False, f"Permutation contains duplicates: {unique_count} unique out of {n}"
    
    return True, "Valid permutation"


def compute_permutation_hash(indices: np.ndarray) -> str:
    """Generates a stable 16-character hexadecimal SHA-256 hash of the permutation indices."""
    if indices is None or len(indices) == 0:
        return "empty"
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def apply_inverse_mapping(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    inverse_indices: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Applies the identical inverse permutation mapping to hard bits and soft LLRs.
    
    CRITICAL RULE:
    Any bit permutation applied to hard bits MUST be applied identically to LLRs.
    Never reorder hard bits and soft bits differently.
    
    Args:
        hard_bits: 1D uint8 array of demodulated bits.
        soft_llrs: Optional 1D float32 array of soft LLRs.
        inverse_indices: 1D integer array of deinterleaving gather indices.
        
    Returns:
        (deinterleaved_hard_bits, deinterleaved_soft_llrs)
    """
    n_perm = len(inverse_indices)
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)
    if len(hard_bits) < n_perm:
        raise ValueError(f"Hard bits length ({len(hard_bits)}) is less than permutation length ({n_perm})")
    
    # Slice exactly to permutation length and reorder
    deinterleaved_hard = hard_bits[:n_perm][inverse_indices].astype(np.uint8)
    
    deinterleaved_soft = None
    if soft_llrs is not None:
        soft_llrs = np.asarray(soft_llrs, dtype=np.float32)
        if len(soft_llrs) < n_perm:
            raise ValueError(f"Soft LLRs length ({len(soft_llrs)}) is less than permutation length ({n_perm})")
        deinterleaved_soft = soft_llrs[:n_perm][inverse_indices].astype(np.float32)
        
    return deinterleaved_hard, deinterleaved_soft


def apply_block_chunked_inverse(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    block_inverse_indices: np.ndarray,
    padding_policy: str = "truncate_tail"
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Applies a fixed block deinterleaving permutation repeatedly across all full blocks in a bitstream.
    
    Args:
        hard_bits: 1D uint8 array
        soft_llrs: Optional 1D float32 array
        block_inverse_indices: 1D integer array of size B (block size)
        padding_policy: 'truncate_tail', 'pad', or 'reject'
        
    Returns:
        (deinterleaved_hard, deinterleaved_soft, remainder_count)
    """
    B = len(block_inverse_indices)
    N = len(hard_bits)
    num_blocks = N // B
    remainder = N % B
    
    if num_blocks == 0:
        if padding_policy == "pad":
            num_blocks = 1
            pad_len = B - N
            hard_padded = np.pad(hard_bits, (0, pad_len), mode='constant', constant_values=0)
            soft_padded = np.pad(soft_llrs, (0, pad_len), mode='constant', constant_values=0.0) if soft_llrs is not None else None
            return hard_padded[block_inverse_indices], soft_padded[block_inverse_indices] if soft_padded is not None else None, remainder
        else:
            raise ValueError(f"Bitstream length {N} is smaller than block size {B}")
            
    usable_len = num_blocks * B
    hard_usable = hard_bits[:usable_len].reshape(num_blocks, B)
    # Vectorized gather across all blocks
    deinterleaved_hard_blocks = hard_usable[:, block_inverse_indices]
    deinterleaved_hard = deinterleaved_hard_blocks.ravel()
    
    deinterleaved_soft = None
    if soft_llrs is not None:
        soft_usable = soft_llrs[:usable_len].reshape(num_blocks, B)
        deinterleaved_soft_blocks = soft_usable[:, block_inverse_indices]
        deinterleaved_soft = deinterleaved_soft_blocks.ravel()
        
    return deinterleaved_hard, deinterleaved_soft, remainder


def compute_permutation_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Computes normalized Hamming distance between two permutations of the same length.
    Returns 0.0 for identical permutations, 1.0 for completely mismatched permutations.
    """
    if len(p1) != len(p2) or len(p1) == 0:
        return 1.0
    return float(np.mean(np.asarray(p1) != np.asarray(p2)))
