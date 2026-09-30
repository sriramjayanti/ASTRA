"""
bit_order.py
Bit-level ordering (MSB-first vs LSB-first within byte and bitstream) utilities.
"""

from typing import List, Union
import numpy as np


def reverse_bits_in_byte(val: int) -> int:
    """Reverse 8 bits in an integer byte: e.g. 0b10000000 -> 0b00000001."""
    res = 0
    for i in range(8):
        if (val >> i) & 1:
            res |= (1 << (7 - i))
    return res


def reverse_bits_in_bytes(byte_data: Union[bytes, bytearray, List[int]]) -> bytes:
    """Reverse bit order within every byte in a byte sequence."""
    return bytes(reverse_bits_in_byte(b) for b in byte_data)


def bits_to_uint(bits: np.ndarray, bit_order: str = "msb") -> int:
    """
    Convert 1D uint8 array of binary bits to an unsigned integer.
    Args:
        bits: 1D array of {0, 1}
        bit_order: 'msb' (MSB first, standard) or 'lsb' (LSB first)
    """
    if len(bits) == 0:
        return 0

    if bit_order.lower() == "lsb":
        bits = bits[::-1]

    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def uint_to_bits(val: int, width_bits: int, bit_order: str = "msb") -> np.ndarray:
    """Convert unsigned integer to 1D binary numpy array of given bit width."""
    bits = np.zeros(width_bits, dtype=np.uint8)
    for i in range(width_bits):
        bits[width_bits - 1 - i] = (val >> i) & 1
    if bit_order.lower() == "lsb":
        bits = bits[::-1]
    return bits


def bits_to_int(bits: np.ndarray, bit_order: str = "msb") -> int:
    """
    Convert 1D uint8 array of binary bits to a signed integer using two's complement.
    """
    if len(bits) == 0:
        return 0

    u_val = bits_to_uint(bits, bit_order=bit_order)
    width = len(bits)
    sign_bit = 1 << (width - 1)

    if u_val & sign_bit:
        return u_val - (1 << width)
    return u_val


def int_to_bits(val: int, width_bits: int, bit_order: str = "msb") -> np.ndarray:
    """Convert signed integer to two's-complement 1D binary numpy array."""
    if val < 0:
        val = (1 << width_bits) + val
    return uint_to_bits(val, width_bits, bit_order=bit_order)


def reverse_bit_stream(bits: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Reverse bits within each block (e.g. within each byte)."""
    n = len(bits)
    out = bits.copy()
    for i in range(0, n, block_size):
        chunk = out[i:min(n, i + block_size)]
        out[i:min(n, i + block_size)] = chunk[::-1]
    return out
