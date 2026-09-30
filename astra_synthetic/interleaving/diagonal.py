"""
Diagonal Matrix Interleaver (Top-Left to Bottom-Right Traversal) for ASTRA Interleaving Engine.
Implements deterministic diagonal matrix reading, permutation indexing, multi-block segmentation, and exact inversion.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .permutation import apply_permutation, invert_permutation, validate_permutation
from .padding import segment_into_blocks, unpad


def compute_diagonal_permutation(
    rows: int,
    columns: int,
    direction: str = "top_left_to_bottom_right",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the forward and inverse index permutations for an R x C diagonal matrix interleaver.

    In 'top_left_to_bottom_right' mode, the matrix is filled row-by-row and read along anti-diagonals
    indexed by d = r + c (d = 0, 1, ..., R+C-2).

    Args:
        rows: Number of matrix rows (R).
        columns: Number of matrix columns (C).
        direction: Traversal direction ('top_left_to_bottom_right').

    Returns:
        tuple (forward_permutation, inverse_permutation).
    """
    if rows < 1 or columns < 1:
        raise ValueError(f"Rows and columns must be >= 1, got rows={rows}, columns={columns}")

    block_size = rows * columns
    perm_indices: list[int] = []

    if direction == "top_left_to_bottom_right":
        # Sum d = r + c ranges from 0 to (rows + columns - 2)
        for d in range(rows + columns - 1):
            r_start = max(0, d - columns + 1)
            r_end = min(rows - 1, d)
            for r in range(r_start, r_end + 1):
                c = d - r
                idx = r * columns + c
                perm_indices.append(idx)
    else:
        raise ValueError(f"Unsupported diagonal direction '{direction}'")

    perm = np.array(perm_indices, dtype=np.int64)
    validate_permutation(perm, block_size)
    inv_perm = invert_permutation(perm)
    return perm, inv_perm


class DiagonalInterleaver:
    """Configurable Diagonal Matrix Interleaver and Reference Deinterleaver."""

    def __init__(
        self,
        rows: int = 8,
        columns: int = 8,
        direction: str = "top_left_to_bottom_right",
        pad_mode: str = "zeros",
    ):
        """Initialize DiagonalInterleaver.

        Args:
            rows: Matrix row count.
            columns: Matrix column count.
            direction: Traversal direction ('top_left_to_bottom_right').
            pad_mode: Padding mode for final block ('zeros', 'ones', 'random').
        """
        self.rows = rows
        self.columns = columns
        self.block_size = rows * columns
        self.direction = direction
        self.pad_mode = pad_mode

        self.permutation, self.inverse_permutation = compute_diagonal_permutation(
            rows=rows,
            columns=columns,
            direction=direction,
        )

    def interleave(
        self,
        bits: np.ndarray,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]], dict[str, Any]]:
        """Interleave input bits block-by-block along matrix diagonals.

        Args:
            bits: 1D NumPy uint8 array of input bits (0 and 1).
            seed: Seed for random padding if applicable.

        Returns:
            tuple (interleaved_bits, padding_bits, pad_length, block_boundaries, parameters)
        """
        if not isinstance(bits, np.ndarray):
            bits = np.asarray(bits, dtype=np.uint8)
        if bits.ndim != 1:
            raise ValueError(f"Input bits must be 1D, got shape {bits.shape}")

        blocks, pad_bits, pad_len, block_meta = segment_into_blocks(
            bits=bits,
            block_size=self.block_size,
            pad_mode=self.pad_mode,
            seed=seed,
        )

        interleaved_blocks = []
        for blk in blocks:
            int_blk = apply_permutation(blk, self.permutation)
            interleaved_blocks.append(int_blk)

        interleaved_arr = (
            np.concatenate(interleaved_blocks) if interleaved_blocks else np.empty(0, dtype=np.uint8)
        )

        params = {
            "rows": self.rows,
            "columns": self.columns,
            "block_size": self.block_size,
            "direction": self.direction,
            "pad_mode": self.pad_mode,
        }

        return interleaved_arr, pad_bits, pad_len, block_meta, params

    def deinterleave(
        self,
        interleaved_bits: np.ndarray,
        original_bit_length: int,
        block_boundaries: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Reference Diagonal Deinterleaver using inverse permutation.

        Args:
            interleaved_bits: Received bit array.
            original_bit_length: Expected original information bit length.
            block_boundaries: Block boundary metadata.

        Returns:
            Decoded original bit array of length `original_bit_length`.
        """
        num_blocks = len(interleaved_bits) // self.block_size
        recovered_blocks = []

        for b_idx in range(num_blocks):
            blk = interleaved_bits[b_idx * self.block_size : (b_idx + 1) * self.block_size]
            deint_blk = apply_permutation(blk, self.inverse_permutation)
            recovered_blocks.append(deint_blk)

        all_recovered = (
            np.concatenate(recovered_blocks) if recovered_blocks else np.empty(0, dtype=np.uint8)
        )
        return unpad(all_recovered, original_bit_length)


def interleave_diagonal(
    bits: np.ndarray,
    rows: int = 8,
    cols: int = 8,
    direction: str = "top_left_to_bottom_right",
    padding_mode: str = "zeros",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Convenience functional interface for diagonal interleaving."""
    interleaver = DiagonalInterleaver(
        rows=rows,
        columns=cols,
        direction=direction,
        pad_mode=padding_mode,
    )
    interleaved_arr, pad_bits, _, block_meta, _ = interleaver.interleave(bits, seed=seed)
    return interleaved_arr, pad_bits, block_meta


def deinterleave_diagonal(
    interleaved_bits: np.ndarray,
    rows: int = 8,
    cols: int = 8,
    direction: str = "top_left_to_bottom_right",
    original_bit_length: int | None = None,
) -> np.ndarray:
    """Convenience functional interface for diagonal reference deinterleaving."""
    interleaver = DiagonalInterleaver(
        rows=rows,
        columns=cols,
        direction=direction,
    )
    orig_len = original_bit_length if original_bit_length is not None else len(interleaved_bits)
    return interleaver.deinterleave(interleaved_bits, original_bit_length=orig_len)

