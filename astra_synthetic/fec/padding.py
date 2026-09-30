"""
Padding and block segmentation utilities for ASTRA FEC Engine.
Tracks exact padding ground truth to prevent loss or corruption of original bits.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def pad_to_multiple(
    bits: np.ndarray,
    multiple: int,
    pad_value: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad a 1D NumPy bit array to the nearest multiple of `multiple`.

    Args:
        bits: 1D NumPy uint8 bit array.
        multiple: Target alignment factor (e.g. 8 for byte alignment, block size for block codes).
        pad_value: Value of padding bits (default: 0).

    Returns:
        tuple of (padded_bits, padding_bits, padding_length).
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
    padding_bits = np.full(pad_len, pad_value, dtype=np.uint8)
    padded_bits = np.concatenate([bits, padding_bits])
    return padded_bits, padding_bits, pad_len


def segment_into_blocks(
    bits: np.ndarray,
    block_size: int,
    pad_final: bool = True,
    pad_value: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, int, list[dict[str, Any]]]:
    """Split a bit array into fixed-size blocks, optionally padding the final block.

    Args:
        bits: 1D NumPy uint8 bit array.
        block_size: Number of bits per block (k).
        pad_final: Whether to zero-pad the last block to exactly `block_size`.
        pad_value: Value for padding bits (default: 0).

    Returns:
        tuple (blocks_list, total_padding_bits, total_padding_length, block_metadata_list)
    """
    if not isinstance(bits, np.ndarray):
        bits = np.asarray(bits, dtype=np.uint8)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    if len(bits) == 0:
        if pad_final:
            pad = np.full(block_size, pad_value, dtype=np.uint8)
            meta = [{
                "block_index": 0,
                "input_start_bit": 0,
                "original_bit_length": 0,
                "padded_bit_length": block_size,
                "padding_length": block_size,
            }]
            return [pad], pad, block_size, meta
        return [], np.empty(0, dtype=np.uint8), 0, []

    blocks = []
    block_meta = []
    all_padding_bits = []
    total_pad_len = 0

    num_full = len(bits) // block_size
    for i in range(num_full):
        start = i * block_size
        chunk = bits[start : start + block_size]
        blocks.append(chunk)
        block_meta.append({
            "block_index": i,
            "input_start_bit": start,
            "original_bit_length": block_size,
            "padded_bit_length": block_size,
            "padding_length": 0,
        })

    rem = len(bits) % block_size
    if rem > 0:
        start = num_full * block_size
        chunk = bits[start:]
        if pad_final:
            pad_len = block_size - rem
            pad_arr = np.full(pad_len, pad_value, dtype=np.uint8)
            padded_chunk = np.concatenate([chunk, pad_arr])
            blocks.append(padded_chunk)
            all_padding_bits.append(pad_arr)
            total_pad_len += pad_len
            block_meta.append({
                "block_index": len(blocks) - 1,
                "input_start_bit": start,
                "original_bit_length": rem,
                "padded_bit_length": block_size,
                "padding_length": pad_len,
            })
        else:
            blocks.append(chunk)
            block_meta.append({
                "block_index": len(blocks) - 1,
                "input_start_bit": start,
                "original_bit_length": rem,
                "padded_bit_length": rem,
                "padding_length": 0,
            })

    padding_arr = (
        np.concatenate(all_padding_bits) if all_padding_bits else np.empty(0, dtype=np.uint8)
    )
    return blocks, padding_arr, total_pad_len, block_meta


def unpad(bits: np.ndarray, original_length: int) -> np.ndarray:
    """Safely unpad a bit array back to its original length.

    Args:
        bits: 1D NumPy bit array.
        original_length: Number of bits in the original unpadded array.

    Returns:
        1D NumPy uint8 array of length `original_length`.
    """
    if len(bits) < original_length:
        raise ValueError(
            f"Cannot unpad: bit array length ({len(bits)}) is smaller than original_length ({original_length})"
        )
    return bits[:original_length]
