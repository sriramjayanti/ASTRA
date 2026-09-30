"""
Block Interleaver (Matrix Row-Write / Column-Read) for ASTRA Interleaving Engine.
Supports configurable rows, columns, write/read order, multi-block segmentation, and exact matrix inversion.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .permutation import apply_permutation, invert_permutation, validate_permutation
from .padding import segment_into_blocks, unpad


def compute_block_permutation(
    rows: int,
    columns: int,
    write_order: str = "row_major",
    read_order: str = "column_major",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the forward and inverse index permutations for an R x C block interleaver.

    Args:
        rows: Number of rows (R).
        columns: Number of columns (C).
        write_order: 'row_major' or 'column_major'.
        read_order: 'column_major' or 'row_major'.

    Returns:
        tuple (forward_permutation, inverse_permutation).
    """
    if rows < 1 or columns < 1:
        raise ValueError(f"Rows and columns must be >= 1 (got rows={rows}, columns={columns})")

    block_size = rows * columns

    if write_order == "row_major" and read_order == "column_major":
        matrix = np.arange(block_size).reshape((rows, columns))
        perm = matrix.T.flatten()
    elif write_order == "column_major" and read_order == "row_major":
        matrix = np.arange(block_size).reshape((columns, rows))
        perm = matrix.T.flatten()
    elif write_order == read_order:
        perm = np.arange(block_size)
    else:
        raise ValueError(f"Unsupported write/read combination: write='{write_order}', read='{read_order}'")

    validate_permutation(perm, block_size)
    inv_perm = invert_permutation(perm)
    return perm, inv_perm


class BlockInterleaver:
    """Configurable Block Interleaver and Reference Deinterleaver."""

    def __init__(
        self,
        rows: int = 16,
        columns: int = 32,
        write_order: str = "row_major",
        read_order: str = "column_major",
        pad_mode: str = "zeros",
    ):
        """Initialize BlockInterleaver with matrix dimensions.

        Args:
            rows: Matrix row count.
            columns: Matrix column count.
            write_order: 'row_major' or 'column_major'.
            read_order: 'column_major' or 'row_major'.
            pad_mode: 'zeros', 'ones', or 'random'.
        """
        self.rows = rows
        self.columns = columns
        self.block_size = rows * columns
        self.write_order = write_order
        self.read_order = read_order
        self.pad_mode = pad_mode

        self.permutation, self.inverse_permutation = compute_block_permutation(
            rows=rows,
            columns=columns,
            write_order=write_order,
            read_order=read_order,
        )

    def interleave(
        self,
        bits: np.ndarray,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]], dict[str, Any]]:
        """Interleave input bits block-by-block using matrix row-write / column-read.

        Args:
            bits: 1D NumPy uint8 array of input bits (0 and 1).
            seed: Seed for random padding if pad_mode='random'.

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
            "write_order": self.write_order,
            "read_order": self.read_order,
            "pad_mode": self.pad_mode,
        }

        return interleaved_arr, pad_bits, pad_len, block_meta, params

    def deinterleave(
        self,
        interleaved_bits: np.ndarray,
        original_bit_length: int,
        block_boundaries: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Reference Block Deinterleaver: applies inverse permutation and removes padding.

        Args:
            interleaved_bits: 1D NumPy uint8 array of interleaved bits.
            original_bit_length: Expected length of original unpadded information bits.
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


def interleave_block(
    bits: np.ndarray,
    rows: int = 16,
    cols: int = 32,
    write_order: str = "row_major",
    read_order: str = "column_major",
    padding_mode: str = "zeros",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Convenience functional interface for block interleaving."""
    interleaver = BlockInterleaver(
        rows=rows,
        columns=cols,
        write_order=write_order,
        read_order=read_order,
        pad_mode=padding_mode,
    )
    interleaved_arr, pad_bits, _, block_meta, _ = interleaver.interleave(bits, seed=seed)
    return interleaved_arr, pad_bits, block_meta


def deinterleave_block(
    interleaved_bits: np.ndarray,
    rows: int = 16,
    cols: int = 32,
    write_order: str = "row_major",
    read_order: str = "column_major",
    original_bit_length: int | None = None,
) -> np.ndarray:
    """Convenience functional interface for block reference deinterleaving."""
    interleaver = BlockInterleaver(
        rows=rows,
        columns=cols,
        write_order=write_order,
        read_order=read_order,
    )
    orig_len = original_bit_length if original_bit_length is not None else len(interleaved_bits)
    return interleaver.deinterleave(interleaved_bits, original_bit_length=orig_len)

