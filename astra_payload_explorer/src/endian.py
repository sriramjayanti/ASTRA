"""
endian.py
Byte-order endianness (big-endian network order vs little-endian) decoding and encoding utilities.
"""

from typing import Union
import numpy as np
from .bit_order import bits_to_uint, bits_to_int, uint_to_bits


def decode_integer_with_endian(
    bits: np.ndarray,
    endian: str = "big",
    signed: bool = False,
    bit_order: str = "msb"
) -> int:
    """
    Decode multi-byte/bit integer supporting big or little endianness.
    """
    if len(bits) == 0:
        return 0

    width = len(bits)

    # For little-endian byte order (when width is multiple of 8)
    if endian.lower() == "little" and width >= 16 and width % 8 == 0:
        n_bytes = width // 8
        reordered_bits = []
        for b_idx in reversed(range(n_bytes)):
            start = b_idx * 8
            reordered_bits.extend(bits[start:start + 8])
        eval_bits = np.array(reordered_bits, dtype=np.uint8)
    else:
        eval_bits = bits

    if signed:
        return bits_to_int(eval_bits, bit_order=bit_order)
    else:
        return bits_to_uint(eval_bits, bit_order=bit_order)


def encode_integer_with_endian(
    val: int,
    width_bits: int,
    endian: str = "big",
    signed: bool = False,
    bit_order: str = "msb"
) -> np.ndarray:
    """Encode an integer value to a binary array with specified endianness."""
    raw_u = val if not signed or val >= 0 else (1 << width_bits) + val
    bits = uint_to_bits(raw_u, width_bits, bit_order=bit_order)

    if endian.lower() == "little" and width_bits >= 16 and width_bits % 8 == 0:
        n_bytes = width_bits // 8
        reordered_bits = []
        for b_idx in reversed(range(n_bytes)):
            start = b_idx * 8
            reordered_bits.extend(bits[start:start + 8])
        return np.array(reordered_bits, dtype=np.uint8)

    return bits


def decode_fixed_point(
    bits: np.ndarray,
    scale: float = 1.0,
    offset: float = 0.0,
    signed: bool = False,
    endian: str = "big"
) -> float:
    """
    Decode scaled fixed-point value: value = (raw_int * scale) + offset.
    """
    raw_int = decode_integer_with_endian(bits, endian=endian, signed=signed)
    return float(raw_int * scale + offset)
