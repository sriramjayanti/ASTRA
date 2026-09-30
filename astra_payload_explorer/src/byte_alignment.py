"""
byte_alignment.py
Bit-to-byte conversion, hex string formatting, and byte array handling.
"""

from typing import Tuple
import numpy as np


def bits_to_bytes(bits: np.ndarray, bit_order: str = "msb") -> bytes:
    """
    Convert 1D uint8 binary bits {0, 1} to Python bytes.
    If bit length is not multiple of 8, pads trailing zeros to complete the last byte.
    """
    if len(bits) == 0:
        return b""

    n_bits = len(bits)
    remainder = n_bits % 8
    pad_len = (8 - remainder) % 8

    if pad_len > 0:
        padded = np.pad(bits, (0, pad_len), mode="constant", constant_values=0)
    else:
        padded = bits

    n_bytes = len(padded) // 8
    byte_matrix = padded.reshape(n_bytes, 8)

    if bit_order.lower() == "msb":
        powers = 2 ** np.arange(7, -1, -1, dtype=np.uint8)
    else:
        powers = 2 ** np.arange(0, 8, 1, dtype=np.uint8)

    byte_values = np.dot(byte_matrix.astype(np.uint8), powers)
    return bytes(byte_values.tolist())


def bytes_to_bits(data: bytes, bit_order: str = "msb") -> np.ndarray:
    """Convert Python bytes to 1D uint8 binary array {0, 1}."""
    bits = []
    for b in data:
        if bit_order.lower() == "msb":
            shifts = (7, 6, 5, 4, 3, 2, 1, 0)
        else:
            shifts = (0, 1, 2, 3, 4, 5, 6, 7)
        for s in shifts:
            bits.append((b >> s) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_hex_str(bits: np.ndarray, bit_order: str = "msb") -> str:
    """Convert 1D binary bits to lowercase hex string without 0x prefix."""
    raw_b = bits_to_bytes(bits, bit_order=bit_order)
    return raw_b.hex().lower()


def hex_to_bits(hex_str: str, bit_order: str = "msb") -> np.ndarray:
    """Convert hex string to 1D uint8 binary array."""
    clean_hex = hex_str.strip().lower().replace("0x", "")
    if len(clean_hex) % 2 != 0:
        clean_hex = "0" + clean_hex
    raw_b = bytes.fromhex(clean_hex)
    return bytes_to_bits(raw_b, bit_order=bit_order)
