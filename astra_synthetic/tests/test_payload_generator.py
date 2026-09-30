"""
Unit tests for ASTRA PayloadGenerator core functionality and bit/byte utilities.
"""

import math
import hashlib
import numpy as np
import pytest

from astra_synthetic.payload.models import (
    PayloadRecord,
    bytes_to_bits,
    bits_to_bytes,
    hex_to_bits,
    bits_to_hex,
    text_to_bits,
    calculate_entropy,
)
from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.payload.validators import (
    ValidationError,
    validate_payload_bits,
    validate_pattern,
    validate_probability,
    validate_hex_string,
    validate_config,
)


class TestBitByteUtilities:
    """Test conversions between bytes, bits, hex, and text."""

    def test_msb_first_byte_to_bits(self):
        # 0xA5 is 10100101 in binary MSB-first
        bits = bytes_to_bits(b"\xA5")
        expected = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        assert np.array_equal(bits, expected)
        assert bits.dtype == np.uint8

    def test_byte_bits_round_trip(self):
        """TEST 5: byte -> bits -> byte round trip"""
        original_bytes = b"Hello ASTRA 123 \x00\xFF\x55\xAA"
        bits = bytes_to_bits(original_bytes)
        recovered_bytes = bits_to_bytes(bits, padding=False)
        assert recovered_bytes == original_bytes

    def test_bits_to_bytes_padding_error(self):
        bits = np.array([1, 0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="not a multiple of 8"):
            bits_to_bytes(bits, padding=False)
        
        # With padding=True, should pad 5 zeros: 10100000 = 0xA0 = 160
        padded_bytes = bits_to_bytes(bits, padding=True)
        assert padded_bytes == bytes([0xA0])

    def test_text_round_trip(self):
        """TEST 6: text -> bits -> bytes -> text round trip"""
        text = "ASTRA SIGNAL TEST FRAME 001 🚀"
        bits = text_to_bits(text, encoding="utf-8")
        recovered_bytes = bits_to_bytes(bits)
        recovered_text = recovered_bytes.decode("utf-8")
        assert recovered_text == text

    def test_hex_conversion(self):
        hex_str = "DEADBEEF"
        bits = hex_to_bits(hex_str)
        assert len(bits) == 32
        recovered_hex = bits_to_hex(bits)
        assert recovered_hex == "DEADBEEF"

    def test_hex_with_0x_and_spaces(self):
        bits = hex_to_bits("  0xDE AD BE EF  ")
        assert len(bits) == 32
        assert bits_to_hex(bits) == "DEADBEEF"


class TestEntropyCalculations:
    """Test Shannon entropy calculation."""

    def test_entropy_all_zeros(self):
        """TEST 10: entropy of all zeros ≈ 0"""
        bits = np.zeros(1024, dtype=np.uint8)
        h = calculate_entropy(bits)
        assert h == 0.0

    def test_entropy_all_ones(self):
        bits = np.ones(1024, dtype=np.uint8)
        h = calculate_entropy(bits)
        assert h == 0.0

    def test_entropy_balanced_random(self):
        """TEST 11: entropy of large balanced random payload ≈ 1"""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=50000, dtype=np.uint8)
        h = calculate_entropy(bits)
        assert abs(h - 1.0) < 0.005


class TestPayloadGeneratorModes:
    """Test each generation mode and parameter constraints."""

    @pytest.fixture
    def generator(self):
        return PayloadGenerator()

    def test_random_bits_length_and_values(self, generator):
        """TEST 1 & TEST 2: random payload length and {0,1} values"""
        rec = generator.generate(payload_type="random_bits", bit_length=1024, seed=42)
        assert rec.bit_length == 1024
        assert len(rec.payload_bits) == 1024
        assert rec.byte_length == 128
        assert len(rec.payload_bytes) == 128
        assert set(np.unique(rec.payload_bits)).issubset({0, 1})
        assert rec.payload_bits.dtype == np.uint8

    def test_random_bytes_mode(self, generator):
        rec = generator.generate(payload_type="random_bytes", byte_length=256, seed=42)
        assert rec.byte_length == 256
        assert rec.bit_length == 2048
        assert len(rec.payload_bits) == 2048
        assert set(np.unique(rec.payload_bits)).issubset({0, 1})

    def test_repeated_pattern_generation(self, generator):
        """TEST 7: repeated pattern generation correct"""
        pattern = "10110011"
        rec = generator.generate(
            payload_type="repeated_pattern",
            pattern=pattern,
            bit_length=20,
        )
        assert rec.bit_length == 20
        expected = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
        assert np.array_equal(rec.payload_bits, expected)
        assert rec.pattern == pattern

    def test_counter_generation(self, generator):
        """TEST 8: counter generation correct"""
        rec = generator.generate(
            payload_type="counter",
            byte_length=512,
            start_value=0,
        )
        assert rec.byte_length == 512
        assert rec.bit_length == 512 * 8
        expected_bytes = bytes(i % 256 for i in range(512))
        assert rec.payload_bytes == expected_bytes

    def test_biased_payload_probability(self, generator):
        """TEST 9: biased payload approximately matches requested probability"""
        target_p1 = 0.05
        rec = generator.generate(
            payload_type="biased_random",
            bit_length=50000,
            probability_one=target_p1,
            seed=1234,
        )
        actual_p1 = np.count_nonzero(rec.payload_bits) / len(rec.payload_bits)
        assert abs(actual_p1 - target_p1) < 0.01

    def test_fixed_hex_mode(self, generator):
        rec = generator.generate(payload_type="fixed_hex", value="CAFEBABE")
        assert rec.bit_length == 32
        assert rec.payload_bytes == bytes.fromhex("CAFEBABE")

    def test_fixed_bits_mode(self, generator):
        rec = generator.generate(payload_type="fixed_bits", value="10110101")
        assert rec.bit_length == 8
        assert np.array_equal(rec.payload_bits, np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8))

    def test_sha256_verification(self, generator):
        """TEST 12: SHA-256 verification"""
        rec = generator.generate(payload_type="random_bits", bit_length=1024, seed=42)
        expected_sha = hashlib.sha256(rec.payload_bytes).hexdigest()
        assert rec.sha256 == expected_sha


class TestValidationAndErrors:
    """TEST 14: invalid configurations raise errors"""

    def test_invalid_pattern(self):
        with pytest.raises(ValidationError):
            validate_pattern("1010201")
        with pytest.raises(ValidationError):
            validate_pattern("")

    def test_invalid_probability(self):
        with pytest.raises(ValidationError):
            validate_probability(1.5)
        with pytest.raises(ValidationError):
            validate_probability(-0.1)

    def test_invalid_hex(self):
        with pytest.raises(ValidationError):
            validate_hex_string("ZZTOP")
        with pytest.raises(ValidationError):
            validate_hex_string("123")  # odd length

    def test_invalid_payload_type(self):
        gen = PayloadGenerator()
        with pytest.raises(ValidationError, match="Unknown payload_type"):
            gen.generate(payload_type="unsupported_quantum_mode")

    def test_invalid_config_probabilities(self):
        bad_config = {
            "payload_generator": {
                "types": {
                    "random_bits": {"enabled": True, "probability": 0.8},
                    "counter": {"enabled": True, "probability": 0.8},
                }
            }
        }
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            validate_config(bad_config)
