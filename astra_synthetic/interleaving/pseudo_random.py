"""
Pseudo-Random Permutation Interleaver for ASTRA Interleaving Engine.
Generates deterministic seed-controlled bit permutations with SHA-256 validation and inverse deinterleaving.
"""

from __future__ import annotations

import hashlib
from typing import Any
import numpy as np

from .permutation import apply_permutation, invert_permutation, validate_permutation
from .padding import segment_into_blocks, unpad


def generate_pseudo_random_permutation(
    block_size: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a reproducible pseudo-random permutation of length `block_size`.

    Args:
        block_size: Number of bits per block (B).
        seed: Deterministic random seed.

    Returns:
        tuple (permutation, inverse_permutation, permutation_sha256).
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(block_size).astype(np.int64)
    validate_permutation(perm, block_size)
    inv_perm = invert_permutation(perm)
    perm_sha = hashlib.sha256(perm.tobytes()).hexdigest()

    return perm, inv_perm, perm_sha


class PseudoRandomInterleaver:
    """Deterministic Pseudo-Random Interleaver and Reference Deinterleaver."""

    def __init__(
        self,
        block_size: int = 256,
        seed: int = 42,
        pad_mode: str = "zeros",
    ):
        """Initialize PseudoRandomInterleaver.

        Args:
            block_size: Permutation block size in bits.
            seed: Master seed for permutation generator.
            pad_mode: Padding mode for final block.
        """
        self.block_size = block_size
        self.seed = seed
        self.pad_mode = pad_mode

        (
            self.permutation,
            self.inverse_permutation,
            self.permutation_sha256,
        ) = generate_pseudo_random_permutation(
            block_size=block_size,
            seed=seed,
        )

    def interleave(
        self,
        bits: np.ndarray,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]], dict[str, Any]]:
        """Interleave input bits block-by-block using pseudo-random index permutation.

        Args:
            bits: 1D NumPy uint8 array of input bits (0 and 1).
            seed: Optional override seed for padding mode.

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
            seed=seed if seed is not None else self.seed,
        )

        interleaved_blocks = []
        for blk in blocks:
            int_blk = apply_permutation(blk, self.permutation)
            interleaved_blocks.append(int_blk)

        interleaved_arr = (
            np.concatenate(interleaved_blocks) if interleaved_blocks else np.empty(0, dtype=np.uint8)
        )

        params = {
            "block_size": self.block_size,
            "seed": self.seed,
            "rng_algorithm": "numpy.random.default_rng(PCG64)",
            "permutation_sha256": self.permutation_sha256,
            "pad_mode": self.pad_mode,
        }

        return interleaved_arr, pad_bits, pad_len, block_meta, params

    def deinterleave(
        self,
        interleaved_bits: np.ndarray,
        original_bit_length: int,
        block_boundaries: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Reference Pseudo-Random Deinterleaver using inverse permutation.

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


def interleave_pseudo_random(
    bits: np.ndarray,
    block_size: int = 256,
    seed: int = 42,
    padding_mode: str = "zeros",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    """Convenience functional interface for pseudo-random interleaving."""
    interleaver = PseudoRandomInterleaver(
        block_size=block_size,
        seed=seed,
        pad_mode=padding_mode,
    )
    interleaved_arr, pad_bits, _, block_meta, _ = interleaver.interleave(bits, seed=seed)
    return interleaved_arr, pad_bits, block_meta, interleaver.permutation


def deinterleave_pseudo_random(
    interleaved_bits: np.ndarray,
    block_size: int = 256,
    seed: int = 42,
    original_bit_length: int | None = None,
) -> np.ndarray:
    """Convenience functional interface for pseudo-random reference deinterleaving."""
    interleaver = PseudoRandomInterleaver(
        block_size=block_size,
        seed=seed,
    )
    orig_len = original_bit_length if original_bit_length is not None else len(interleaved_bits)
    return interleaver.deinterleave(interleaved_bits, original_bit_length=orig_len)

