"""
block.py
Rectangular block interleaver and deinterleaver family.
Supports row-to-column and column-to-row matrix orientations with tail truncation/padding policies.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from .models import PermutationMapping, InterleaverFamily
from .permutation import (
    invert_permutation,
    validate_permutation,
    compute_permutation_hash,
    apply_block_chunked_inverse,
    apply_inverse_mapping
)


def create_block_mapping(
    rows: int,
    cols: int,
    orientation: str = "row_to_column"
) -> PermutationMapping:
    """
    Creates a PermutationMapping for a rectangular block of size (rows x cols).
    
    Orientations:
      - 'row_to_column':
          Interleaver writes row-wise, reads column-wise.
          Deinterleaver restores by writing column-wise, reading row-wise.
      - 'column_to_row':
          Interleaver writes column-wise, reads row-wise.
          Deinterleaver restores by writing row-wise, reading column-wise.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Block rows ({rows}) and cols ({cols}) must be positive integers")
        
    length = rows * cols
    grid = np.arange(length, dtype=np.int64)
    
    if orientation == "row_to_column":
        # Interleaver: write row-by-row (R, C), read col-by-col (transpose to C, R and flatten)
        forward_indices = grid.reshape((rows, cols)).T.ravel()
    elif orientation == "column_to_row":
        # Interleaver: write col-by-col, read row-by-row
        forward_indices = grid.reshape((cols, rows)).T.ravel()
    else:
        raise ValueError(f"Unknown block orientation: {orientation}. Supported: 'row_to_column', 'column_to_row'")
        
    inverse_indices = invert_permutation(forward_indices)
    
    mapping = PermutationMapping(
        forward_indices=forward_indices,
        inverse_indices=inverse_indices,
        length=length,
        family=InterleaverFamily.BLOCK.value,
        parameters={
            "rows": int(rows),
            "cols": int(cols),
            "block_size": int(length),
            "orientation": orientation
        },
        mapping_hash=compute_permutation_hash(inverse_indices)
    )
    
    is_valid, reason = validate_permutation(inverse_indices, expected_length=length)
    if not is_valid:
        raise ValueError(f"Invalid block permutation generated: {reason}")
        
    return mapping


def interleave_block(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    rows: int,
    cols: int,
    orientation: str = "row_to_column",
    padding_policy: str = "truncate_tail"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Interleaves a stream using rectangular block permutation.
    Used for synthetic test vectors and transmitter simulation.
    """
    mapping = create_block_mapping(rows, cols, orientation)
    B = mapping.length
    N = len(hard_bits)
    num_blocks = N // B
    if num_blocks == 0:
        if padding_policy == "pad":
            pad_len = B - N
            hard_padded = np.pad(hard_bits, (0, pad_len), mode='constant', constant_values=0)
            soft_padded = np.pad(soft_llrs, (0, pad_len), mode='constant', constant_values=0.0) if soft_llrs is not None else None
            return hard_padded[mapping.forward_indices], soft_padded[mapping.forward_indices] if soft_padded is not None else None
        else:
            raise ValueError(f"Bitstream length {N} is less than block size {B}")
            
    usable_len = num_blocks * B
    hard_usable = hard_bits[:usable_len].reshape(num_blocks, B)
    interleaved_hard = hard_usable[:, mapping.forward_indices].ravel()
    
    interleaved_soft = None
    if soft_llrs is not None:
        soft_usable = soft_llrs[:usable_len].reshape(num_blocks, B)
        interleaved_soft = soft_usable[:, mapping.forward_indices].ravel()
        
    return interleaved_hard, interleaved_soft


def deinterleave_block(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    rows: int,
    cols: int,
    orientation: str = "row_to_column",
    padding_policy: str = "truncate_tail"
) -> Tuple[np.ndarray, Optional[np.ndarray], PermutationMapping, int]:
    """
    Deinterleaves a stream using rectangular block permutation.
    
    Returns:
        (deinterleaved_hard_bits, deinterleaved_soft_llrs, mapping, remainder_count)
    """
    mapping = create_block_mapping(rows, cols, orientation)
    deinterleaved_hard, deinterleaved_soft, remainder = apply_block_chunked_inverse(
        hard_bits=hard_bits,
        soft_llrs=soft_llrs,
        block_inverse_indices=mapping.inverse_indices,
        padding_policy=padding_policy
    )
    return deinterleaved_hard, deinterleaved_soft, mapping, remainder
