"""
Unit tests for bit grouping, Gray coding, and mapping tables in ASTRA Engine 5.
"""

import numpy as np
import pytest

from astra_synthetic.modulation.bit_mapping import (
    PSK8_GRAY_INVERSE,
    PSK8_GRAY_MAP,
    QAM16_AXIS_GRAY,
    QAM16_AXIS_INVERSE,
    QAM64_AXIS_GRAY,
    QAM64_AXIS_INVERSE,
    QPSK_GRAY_INVERSE,
    QPSK_GRAY_MAP,
    binary_symbols_to_bits,
    binary_to_gray,
    gray_to_binary,
    pad_and_group_bits,
)


class TestBitMapping:
    """Test suite for bit packing, grouping, and Gray coding."""

    def test_binary_gray_conversion_roundtrip(self):
        """Verify binary <-> Gray conversions are inverse bijections for 0..255."""
        for val in range(256):
            gray = binary_to_gray(val)
            recovered = gray_to_binary(gray)
            assert recovered == val

    def test_pad_and_group_bits_exact_multiple(self):
        """Verify grouping with input length multiple of bits_per_symbol (no padding)."""
        bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        # 2 bits per symbol: [1,0] -> 2, [1,1] -> 3, [0,0] -> 0
        syms, pad_bits, pad_len, mapped_len = pad_and_group_bits(bits, bits_per_symbol=2)
        np.testing.assert_array_equal(syms, [2, 3, 0])
        assert pad_len == 0
        assert len(pad_bits) == 0
        assert mapped_len == 6

        # Roundtrip unpack
        unpacked = binary_symbols_to_bits(syms, bits_per_symbol=2, original_bit_length=6)
        np.testing.assert_array_equal(unpacked, bits)

    def test_pad_and_group_bits_with_padding(self):
        """Verify zero-padding at the end when input length is not divisible by bits_per_symbol."""
        bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)  # 5 bits, k=3 -> needs 1 pad bit -> [1,0,1], [1,0,0]
        syms, pad_bits, pad_len, mapped_len = pad_and_group_bits(bits, bits_per_symbol=3, pad_mode="zeros")
        np.testing.assert_array_equal(syms, [5, 4])
        assert pad_len == 1
        assert len(pad_bits) == 1
        assert pad_bits[0] == 0
        assert mapped_len == 6

        # Roundtrip unpack with original length trimming
        unpacked = binary_symbols_to_bits(syms, bits_per_symbol=3, original_bit_length=5)
        np.testing.assert_array_equal(unpacked, bits)

    def test_qpsk_gray_mapping_bijections(self):
        """Verify QPSK Gray mapping table has 4 unique points and reversible."""
        assert len(QPSK_GRAY_MAP) == 4
        assert set(QPSK_GRAY_MAP.keys()) == {0, 1, 2, 3}
        assert set(QPSK_GRAY_MAP.values()) == {0, 1, 2, 3}
        for k, v in QPSK_GRAY_MAP.items():
            assert QPSK_GRAY_INVERSE[v] == k

    def test_8psk_gray_mapping_bijections(self):
        """Verify 8-PSK Gray mapping table has 8 unique points and reversible."""
        assert len(PSK8_GRAY_MAP) == 8
        assert set(PSK8_GRAY_MAP.keys()) == set(range(8))
        assert set(PSK8_GRAY_MAP.values()) == set(range(8))
        for k, v in PSK8_GRAY_MAP.items():
            assert PSK8_GRAY_INVERSE[v] == k

    def test_qam_axis_gray_mappings(self):
        """Verify 16-QAM and 64-QAM axis Gray mapping tables."""
        assert len(QAM16_AXIS_GRAY) == 4
        for k, v in QAM16_AXIS_GRAY.items():
            assert QAM16_AXIS_INVERSE[v] == k

        assert len(QAM64_AXIS_GRAY) == 8
        for k, v in QAM64_AXIS_GRAY.items():
            assert QAM64_AXIS_INVERSE[v] == k
