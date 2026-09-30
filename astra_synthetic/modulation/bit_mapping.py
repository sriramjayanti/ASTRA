"""
Bit grouping, Gray coding, and explicit mapping table definitions for ASTRA Engine 5.
Maintains MSB-first bit convention and reversible symbol-to-bit mapping tables.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def pad_and_group_bits(
    bits: np.ndarray,
    bits_per_symbol: int,
    pad_mode: str = "zeros",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Pad input bitstream if needed and group into integer symbol values.

    Args:
        bits: 1D NumPy uint8 array of bits (0 and 1).
        bits_per_symbol: Number of bits per symbol (k >= 1).
        pad_mode: 'zeros' or 'ones'.

    Returns:
        tuple (symbol_indices_binary, padding_bits, padding_length, mapped_bit_length)
    """
    if not isinstance(bits, np.ndarray):
        bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError(f"Input bits must be 1D, got shape {bits.shape}")
    if bits_per_symbol < 1:
        raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol}")

    orig_len = len(bits)
    rem = orig_len % bits_per_symbol
    pad_len = (bits_per_symbol - rem) % bits_per_symbol

    if pad_len > 0:
        pad_val = 1 if pad_mode == "ones" else 0
        pad_bits = np.full(pad_len, pad_val, dtype=np.uint8)
        padded_bits = np.concatenate([bits, pad_bits])
    else:
        pad_bits = np.empty(0, dtype=np.uint8)
        padded_bits = bits

    mapped_len = len(padded_bits)
    num_symbols = mapped_len // bits_per_symbol

    # Reshape and compute integer value (MSB-first)
    bit_matrix = padded_bits.reshape((num_symbols, bits_per_symbol))
    powers_of_two = 2 ** np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    binary_symbols = np.dot(bit_matrix, powers_of_two).astype(np.int64)

    return binary_symbols, pad_bits, pad_len, mapped_len


def binary_symbols_to_bits(
    symbols: np.ndarray,
    bits_per_symbol: int,
    original_bit_length: int | None = None,
) -> np.ndarray:
    """Unpack integer symbol indices back into 1D bitstream (MSB-first) and remove padding.

    Args:
        symbols: 1D NumPy array of symbol integers.
        bits_per_symbol: Number of bits per symbol.
        original_bit_length: Length of original unpadded bitstream.

    Returns:
        1D NumPy uint8 bit array.
    """
    if len(symbols) == 0:
        return np.empty(0, dtype=np.uint8)

    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    bit_matrix = (symbols[:, None] >> shifts) & 1
    unpacked_bits = bit_matrix.flatten().astype(np.uint8)

    if original_bit_length is not None and original_bit_length < len(unpacked_bits):
        return unpacked_bits[:original_bit_length]
    return unpacked_bits


def binary_to_gray(val: int) -> int:
    """Convert standard binary integer to Gray code."""
    return val ^ (val >> 1)


def gray_to_binary(gray: int) -> int:
    """Convert Gray code integer to standard binary integer."""
    bin_val = gray
    mask = gray >> 1
    while mask != 0:
        bin_val ^= mask
        mask >>= 1
    return bin_val


# --------------------------------------------------------------------------
# Explicit Standard Gray Mapping Tables
# --------------------------------------------------------------------------

# QPSK Gray Mapping:
# 00 (0) -> ( 1 + 1j)/sqrt(2) [index 0]
# 01 (1) -> (-1 + 1j)/sqrt(2) [index 1]
# 11 (3) -> (-1 - 1j)/sqrt(2) [index 2]
# 10 (2) -> ( 1 - 1j)/sqrt(2) [index 3]
QPSK_GRAY_MAP = {
    0: 0,  # 00 -> Point 0 (+1, +1)
    1: 1,  # 01 -> Point 1 (-1, +1)
    3: 2,  # 11 -> Point 2 (-1, -1)
    2: 3,  # 10 -> Point 3 (+1, -1)
}
QPSK_GRAY_INVERSE = {v: k for k, v in QPSK_GRAY_MAP.items()}

# 8PSK Gray Mapping (adjacent phase order):
# 000 (0) -> phase 0 ( 0 deg)
# 001 (1) -> phase 1 ( 45 deg)
# 011 (3) -> phase 2 ( 90 deg)
# 010 (2) -> phase 3 (135 deg)
# 110 (6) -> phase 4 (180 deg)
# 111 (7) -> phase 5 (225 deg)
# 101 (5) -> phase 6 (270 deg)
# 100 (4) -> phase 7 (315 deg)
PSK8_GRAY_MAP = {
    0: 0,
    1: 1,
    3: 2,
    2: 3,
    6: 4,
    7: 5,
    5: 6,
    4: 7,
}
PSK8_GRAY_INVERSE = {v: k for k, v in PSK8_GRAY_MAP.items()}

# 16-QAM 2-bit Axis Gray Mapping:
# 00 (0) -> -3
# 01 (1) -> -1
# 11 (3) -> +1
# 10 (2) -> +3
QAM16_AXIS_GRAY = {
    0: -3.0,
    1: -1.0,
    3: 1.0,
    2: 3.0,
}
QAM16_AXIS_INVERSE = {
    -3.0: 0,
    -1.0: 1,
    1.0: 3,
    3.0: 2,
}

# 64-QAM 3-bit Axis Gray Mapping:
# 000 (0) -> -7
# 001 (1) -> -5
# 011 (3) -> -3
# 010 (2) -> -1
# 110 (6) -> +1
# 111 (7) -> +3
# 101 (5) -> +5
# 100 (4) -> +7
QAM64_AXIS_GRAY = {
    0: -7.0,
    1: -5.0,
    3: -3.0,
    2: -1.0,
    6: 1.0,
    7: 3.0,
    5: 5.0,
    4: 7.0,
}
QAM64_AXIS_INVERSE = {
    -7.0: 0,
    -5.0: 1,
    -3.0: 3,
    -1.0: 2,
    1.0: 6,
    3.0: 7,
    5.0: 5,
    7.0: 4,
}
