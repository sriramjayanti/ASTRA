"""
Padding and block segmentation utilities for ASTRA Interleaving Engine.
Supports deterministic zero, one, and seeded pseudo-random padding modes with exact boundary telemetry.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def pad_to_multiple(
    bits: np.ndarray,
    multiple: int,
    pad_mode: str = "zeros",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad a bit array to the nearest multiple of `multiple`.

    Args:
        bits: 1D NumPy uint8 array.
        multiple: Required block alignment factor.
        pad_mode: 'zeros', 'ones', or 'random'.
        seed: Optional seed for random padding mode.

    Returns:
        tuple (padded_bits, padding_bits, padding_length).
    """
    if not isinstance(bits, np.ndarray):
        bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError(f"Bits must be 1D, got shape {bits.shape}")
    if multiple < 1:
        raise ValueError(f"Multiple must be >= 1, got {multiple}")

    rem = len(bits) % multiple
    if rem == 0:
        return bits, np.empty(0, dtype=np.uint8), 0

    pad_len = multiple - rem
    if pad_mode == "zeros":
        pad_bits = np.zeros(pad_len, dtype=np.uint8)
    elif pad_mode == "ones":
        pad_bits = np.ones(pad_len, dtype=np.uint8)
    elif pad_mode == "random":
        rng = np.random.default_rng(seed)
        pad_bits = rng.integers(0, 2, size=pad_len, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown pad_mode '{pad_mode}'. Supported: zeros, ones, random")

    padded_bits = np.concatenate([bits, pad_bits])
    return padded_bits, pad_bits, pad_len


def segment_into_blocks(
    bits: np.ndarray,
    block_size: int,
    pad_mode: str = "zeros",
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, int, list[dict[str, Any]]]:
    """Split a bit array into fixed-size blocks, optionally padding the final block.

    Args:
        bits: 1D NumPy uint8 array.
        block_size: Block size in bits (B).
        pad_mode: Padding mode for final short block.
        seed: Seed for random padding mode.

    Returns:
        tuple (blocks_list, total_padding_bits, total_padding_length, block_boundaries_list).
    """
    if not isinstance(bits, np.ndarray):
        bits = np.asarray(bits, dtype=np.uint8)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    if len(bits) == 0:
        if pad_mode != "none":
            pad = np.zeros(block_size, dtype=np.uint8)
            meta = [{
                "block_id": 0,
                "input_start": 0,
                "input_length": 0,
                "padded_length": block_size,
                "padding_length": block_size,
                "output_start": 0,
                "output_length": block_size,
            }]
            return [pad], pad, block_size, meta
        return [], np.empty(0, dtype=np.uint8), 0, []

    blocks = []
    block_meta = []
    all_padding_bits = []
    total_pad_len = 0

    num_full = len(bits) // block_size
    current_out_offset = 0

    for i in range(num_full):
        start = i * block_size
        chunk = bits[start : start + block_size]
        blocks.append(chunk)
        block_meta.append({
            "block_id": i,
            "input_start": start,
            "input_length": block_size,
            "padded_length": block_size,
            "padding_length": 0,
            "output_start": current_out_offset,
            "output_length": block_size,
        })
        current_out_offset += block_size

    rem = len(bits) % block_size
    if rem > 0:
        start = num_full * block_size
        chunk = bits[start:]
        pad_len = block_size - rem
        if pad_mode == "zeros":
            pad_arr = np.zeros(pad_len, dtype=np.uint8)
        elif pad_mode == "ones":
            pad_arr = np.ones(pad_len, dtype=np.uint8)
        elif pad_mode == "random":
            rng = np.random.default_rng(seed)
            pad_arr = rng.integers(0, 2, size=pad_len, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown pad_mode '{pad_mode}'")

        padded_chunk = np.concatenate([chunk, pad_arr])
        blocks.append(padded_chunk)
        all_padding_bits.append(pad_arr)
        total_pad_len += pad_len

        block_meta.append({
            "block_id": len(blocks) - 1,
            "input_start": start,
            "input_length": rem,
            "padded_length": block_size,
            "padding_length": pad_len,
            "output_start": current_out_offset,
            "output_length": block_size,
        })
        current_out_offset += block_size

    padding_arr = (
        np.concatenate(all_padding_bits) if all_padding_bits else np.empty(0, dtype=np.uint8)
    )
    return blocks, padding_arr, total_pad_len, block_meta


def unpad(bits: np.ndarray, original_length: int) -> np.ndarray:
    """Trim padding from a deinterleaved bit array back to original length."""
    if len(bits) < original_length:
        raise ValueError(f"Cannot unpad: bit length ({len(bits)}) is shorter than original_length ({original_length})")
    return bits[:original_length]
