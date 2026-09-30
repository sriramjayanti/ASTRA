"""
test_endian.py
Unit tests for bit-to-byte conversion, big/little endian integers, two's complement, and fixed-point parsing.
"""

import pytest
import numpy as np

from astra_payload_explorer.src.byte_alignment import (
    bits_to_bytes,
    bytes_to_bits,
    bits_to_hex_str,
    hex_to_bits
)
from astra_payload_explorer.src.bit_order import (
    bits_to_uint,
    bits_to_int,
    reverse_bits_in_byte
)
from astra_payload_explorer.src.endian import (
    decode_integer_with_endian,
    decode_fixed_point
)


def test_1_bits_to_bytes_msb():
    """TEST 1: bits_to_bytes MSB-first conversion."""
    # 0b11101011 0b10010000 = 0xEB 0x90
    bits = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
    data = bits_to_bytes(bits, bit_order="msb")
    assert data == b"\xeb\x90"
    assert bits_to_hex_str(bits).lower() == "eb90"


def test_2_byte_round_trip():
    """TEST 2: bytes to bits and back round-trip."""
    orig_bytes = b"Hello, ASTRA!"
    bits = bytes_to_bits(orig_bytes)
    recovered = bits_to_bytes(bits)
    assert orig_bytes == recovered


def test_3_uint_big_endian():
    """TEST 3: unsigned integer decoding in big-endian."""
    # 0x0102 = 258 in 16-bit big-endian
    bits = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
    val = decode_integer_with_endian(bits, endian="big", signed=False)
    assert val == 258


def test_4_uint_little_endian():
    """TEST 4: unsigned integer decoding in little-endian."""
    # Bytes: 0x02 0x01 in little-endian = 0x0102 = 258
    # Byte 0: 0x02 (00000010), Byte 1: 0x01 (00000001)
    bits = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
    val = decode_integer_with_endian(bits, endian="little", signed=False)
    assert val == 258


def test_5_signed_twos_complement():
    """TEST 5: signed two's complement integer decoding."""
    # 8-bit -5 = 0b11111011 (251)
    bits = np.array([1, 1, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
    val = decode_integer_with_endian(bits, signed=True)
    assert val == -5

    # 16-bit -1000 = 0b11111100 00011000 = 0xFC18
    bits_16 = hex_to_bits("FC18")
    val_16 = decode_integer_with_endian(bits_16, signed=True)
    assert val_16 == -1000


def test_6_fixed_point_parsing():
    """TEST 6: fixed-point scaled value parsing."""
    # Raw int 350 * 0.1 - 10.0 = 25.0
    bits = hex_to_bits("015E") # 350 in hex
    val = decode_fixed_point(bits, scale=0.1, offset=-10.0)
    assert abs(val - 25.0) < 1e-4
