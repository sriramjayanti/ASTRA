"""
Unit tests for ASTRA Header Generator and bit/integer conversions.
Validates MSB-first field packing, sequence increments/wrapping, payload length encoding, and range bounds.
"""

import numpy as np
import pytest

from astra_synthetic.payload.models import PayloadRecord
from astra_synthetic.framing.header import (
    int_to_bits,
    bits_to_int,
    HeaderGenerator,
)


class TestIntBitConversion:
    """Test integer to MSB-first bit array conversions."""

    def test_msb_first_int_to_bits(self):
        """TEST 15: bit/byte conversions preserve MSB-first convention"""
        # Value 5 in 4 bits is 0101
        bits = int_to_bits(5, width=4)
        assert np.array_equal(bits, np.array([0, 1, 0, 1], dtype=np.uint8))
        assert bits_to_int(bits) == 5

        # Value 0x1A2B in 16 bits
        val = 0x1A2B
        bits16 = int_to_bits(val, width=16)
        assert len(bits16) == 16
        assert bits_to_int(bits16) == val

    def test_field_overflow_raises_error(self):
        """TEST 18 & 19: invalid field width or value too large raises error"""
        with pytest.raises(ValueError, match="cannot fit in 4 bits"):
            int_to_bits(16, width=4)  # 4 bits can hold 0-15

        with pytest.raises(ValueError, match="cannot fit in 8 bits"):
            int_to_bits(256, width=8)

        with pytest.raises(ValueError, match="Bit width must be >= 1"):
            int_to_bits(0, width=0)

        with pytest.raises(ValueError, match="cannot fit in 4 bits"):
            int_to_bits(-1, width=4)


class TestHeaderGenerator:
    """Test structured header construction and sequence tracking."""

    @pytest.fixture
    def dummy_payload(self):
        bits = np.ones(1024, dtype=np.uint8)
        return PayloadRecord(
            payload_id="payload_000001",
            payload_type="random_bits",
            payload_bits=bits,
            payload_bytes=bytes([0xFF] * 128),
            bit_length=1024,
            byte_length=128,
        )

    def test_header_field_encoding(self, dummy_payload):
        """TEST 4: header field encoding correct"""
        gen = HeaderGenerator()
        header_bits, resolved_fields, offsets = gen.build_header(
            payload_record=dummy_payload,
            sequence_number=5,
        )

        assert len(header_bits) == 56
        assert resolved_fields["version"] == 1
        assert resolved_fields["frame_type"] == 0
        assert resolved_fields["sequence_number"] == 5
        assert resolved_fields["payload_length"] == 1024
        assert resolved_fields["flags"] == 0

        # Verify exact field slices
        ver_bits = header_bits[offsets["version"]["start"] : offsets["version"]["start"] + offsets["version"]["length"]]
        assert bits_to_int(ver_bits) == 1

        seq_bits = header_bits[offsets["sequence_number"]["start"] : offsets["sequence_number"]["start"] + offsets["sequence_number"]["length"]]
        assert bits_to_int(seq_bits) == 5

    def test_payload_length_field_correctness(self, dummy_payload):
        """TEST 5: payload length field correct"""
        gen = HeaderGenerator()
        header_bits, resolved_fields, offsets = gen.build_header(
            payload_record=dummy_payload,
            sequence_number=1,
        )
        assert resolved_fields["payload_length"] == 1024
        len_bits = header_bits[offsets["payload_length"]["start"] : offsets["payload_length"]["start"] + offsets["payload_length"]["length"]]
        assert bits_to_int(len_bits) == 1024

    def test_sequence_number_wrapping(self, dummy_payload):
        """TEST 6 & TEST 7: sequence increments and wraps at maximum"""
        schema = {
            "fields": {
                "sequence_number": {"bits": 4, "source": "auto_increment"},
            }
        }
        gen = HeaderGenerator(schema)

        # 4 bits wraps at 16
        _, f0, _ = gen.build_header(dummy_payload, sequence_number=0)
        assert f0["sequence_number"] == 0

        _, f15, _ = gen.build_header(dummy_payload, sequence_number=15)
        assert f15["sequence_number"] == 15

        _, f16, _ = gen.build_header(dummy_payload, sequence_number=16)
        assert f16["sequence_number"] == 0  # 16 % 16 == 0

        _, f17, _ = gen.build_header(dummy_payload, sequence_number=17)
        assert f17["sequence_number"] == 1
