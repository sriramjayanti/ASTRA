"""
helical.py
Helical / Diagonal finite permutation interleaver and deinterleaver family.
Traverses a 2D rectangular buffer diagonally with configurable row counts, step offsets, and directions.
"""

from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np
from .models import PermutationMapping, InterleaverFamily
from .permutation import (
    invert_permutation,
    validate_permutation,
    compute_permutation_hash,
    apply_block_chunked_inverse
)


def create_helical_mapping(
    rows: int,
    cols: int,
    step: int = 1,
    orientation: str = "row_diagonal"
) -> PermutationMapping:
    """
    Creates a finite block helical / diagonal permutation mapping.
    
    Grid size: R rows, C cols, total length = R * C.
    
    Orientations:
      - 'row_diagonal':
          Interleaver writes row-wise, reads along diagonals offset by step s.
          Deinterleaver reverses this mapping.
      - 'diagonal_row':
          Interleaver writes along diagonals, reads row-wise.
          Deinterleaver reverses this mapping.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Helical rows ({rows}) and cols ({cols}) must be positive")
    if step <= 0:
        raise ValueError(f"Helical step ({step}) must be positive")
        
    length = rows * cols
    
    # Generate read indices for row_diagonal:
    # For each diagonal d (0 to C-1) and each row r (0 to R-1):
    # col = (d + r * step) % C
    # linear_idx = r * C + col
    read_indices = []
    for d in range(cols):
        for r in range(rows):
            c = (d + r * step) % cols
            read_indices.append(r * cols + c)
            
    read_indices = np.array(read_indices, dtype=np.int64)
    
    if orientation == "row_diagonal":
        forward_indices = read_indices
    elif orientation == "diagonal_row":
        forward_indices = invert_permutation(read_indices)
    else:
        raise ValueError(f"Unknown helical orientation: {orientation}. Supported: 'row_diagonal', 'diagonal_row'")
        
    inverse_indices = invert_permutation(forward_indices)
    
    # Validate permutation bijection
    is_valid, reason = validate_permutation(inverse_indices, expected_length=length)
    if not is_valid:
        raise ValueError(f"Helical configuration (R={rows}, C={cols}, step={step}) is invalid: {reason}")
        
    mapping = PermutationMapping(
        forward_indices=forward_indices,
        inverse_indices=inverse_indices,
        length=length,
        family=InterleaverFamily.HELICAL.value,
        parameters={
            "rows": int(rows),
            "cols": int(cols),
            "step": int(step),
            "block_size": int(length),
            "orientation": orientation
        },
        mapping_hash=compute_permutation_hash(inverse_indices)
    )
    return mapping


def interleave_helical(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    rows: int,
    cols: int,
    step: int = 1,
    orientation: str = "row_diagonal"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Interleaves bitstream with helical permutation.
    """
    mapping = create_helical_mapping(rows, cols, step, orientation)
    B = mapping.length
    N = len(hard_bits)
    num_blocks = N // B
    if num_blocks == 0:
        raise ValueError(f"Bitstream length {N} is less than helical block size {B}")
        
    usable_len = num_blocks * B
    hard_usable = hard_bits[:usable_len].reshape(num_blocks, B)
    interleaved_hard = hard_usable[:, mapping.forward_indices].ravel()
    
    interleaved_soft = None
    if soft_llrs is not None:
        soft_usable = soft_llrs[:usable_len].reshape(num_blocks, B)
        interleaved_soft = soft_usable[:, mapping.forward_indices].ravel()
        
    return interleaved_hard, interleaved_soft


def deinterleave_helical(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    rows: int,
    cols: int,
    step: int = 1,
    orientation: str = "row_diagonal",
    padding_policy: str = "truncate_tail"
) -> Tuple[np.ndarray, Optional[np.ndarray], PermutationMapping, int]:
    """
    Deinterleaves bitstream using helical permutation.
    """
    mapping = create_helical_mapping(rows, cols, step, orientation)
    deint_hard, deint_soft, remainder = apply_block_chunked_inverse(
        hard_bits=hard_bits,
        soft_llrs=soft_llrs,
        block_inverse_indices=mapping.inverse_indices,
        padding_policy=padding_policy
    )
    return deint_hard, deint_soft, mapping, remainder
