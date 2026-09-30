"""
Unit tests for ASTRA CRC Engine.
Validates CRC-8, CRC-16-CCITT, CRC-32 against standard vectors, error detection on bit flips, and profile handling.
"""

import numpy as np
import pytest

from astra_synthetic.payload.models import text_to_bits, bytes_to_bits
from astra_synthetic.framing.crc import (
    CRCProfile,
    compute_crc,
    verify_crc,
    get_crc_profile,
    CRC_PROFILES,
)


class TestCRCEngine:
    """Test CRC computations and corruption detection."""

    def test_crc_profiles_registered(self):
        assert "crc8" in CRC_PROFILES
        assert "crc16_ccitt" in CRC_PROFILES
        assert "crc32" in CRC_PROFILES

        p16 = get_crc_profile("crc16_ccitt")
        assert p16.width == 16
        assert p16.poly == 0x1021

    def test_crc16_ccitt_standard_vector(self):
        """Standard ASCII vector '123456789' for CRC-16-CCITT (False: init=0xFFFF, poly=0x1021) is 0x29B1."""
        bits = text_to_bits("123456789")
        crc_val, crc_bits = compute_crc(bits, crc_type="crc16_ccitt")
        assert crc_val == 0x29B1
        assert len(crc_bits) == 16
        assert verify_crc(bits, crc_bits, crc_type="crc16_ccitt")

    def test_crc8_standard_vector(self):
        """Standard vector for CRC-8/ITU over '123456789' is 0xF4."""
        bits = text_to_bits("123456789")
        crc_val, crc_bits = compute_crc(bits, crc_type="crc8")
        assert crc_val == 0xF4
        assert len(crc_bits) == 8
        assert verify_crc(bits, crc_bits, crc_type="crc8")

    def test_crc32_standard_vector(self):
        """Standard IEEE 802.3 CRC-32 for '123456789' is 0xCBF43926."""
        bits = text_to_bits("123456789")
        crc_val, crc_bits = compute_crc(bits, crc_type="crc32")
        assert crc_val == 0xCBF43926
        assert len(crc_bits) == 32
        assert verify_crc(bits, crc_bits, crc_type="crc32")

    def test_crc_validation_success(self):
        """TEST 8: CRC validates correctly"""
        data = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
        _, crc_bits = compute_crc(data, crc_type="crc16_ccitt")
        assert verify_crc(data, crc_bits, crc_type="crc16_ccitt") is True

    def test_single_bit_flip_fails_crc(self):
        """TEST 9: single-bit corruption causes CRC failure"""
        data = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
        _, crc_bits = compute_crc(data, crc_type="crc16_ccitt")
        
        # Corrupt 1 bit at index 4
        corrupted_data = data.copy()
        corrupted_data[4] ^= 1

        assert verify_crc(corrupted_data, crc_bits, crc_type="crc16_ccitt") is False

    def test_single_crc_bit_flip_fails(self):
        data = np.array([0, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        _, crc_bits = compute_crc(data, crc_type="crc16_ccitt")
        
        corrupted_crc = crc_bits.copy()
        corrupted_crc[0] ^= 1
        assert verify_crc(data, corrupted_crc, crc_type="crc16_ccitt") is False
