"""
Unit tests for ASTRA Concatenated FEC Encoder and Decoder.
Validates two-stage encoding (RS outer + Conv inner), intermediate ground truth, and reverse reference decoding.
"""

import numpy as np
import pytest

from astra_synthetic.fec.reed_solomon import ReedSolomonCode
from astra_synthetic.fec.convolutional import ConvolutionalCode
from astra_synthetic.fec.concatenated import ConcatenatedCode


class TestConcatenatedCode:
    """Test two-stage concatenated coding and reverse decoding."""

    def test_concatenated_clean_round_trip(self):
        """TEST 8: Concatenated clean round-trip."""
        outer_rs = ReedSolomonCode(n=64, k=48)
        inner_conv = ConvolutionalCode(constraint_length=3, generators=[0o7, 0o5])

        concat = ConcatenatedCode(outer_rs=outer_rs, inner_conv=inner_conv)

        rng = np.random.default_rng(42)
        # 300 bits -> fits in 1 RS block of 48 bytes (384 bits)
        in_bits = rng.integers(0, 2, size=300, dtype=np.uint8)

        (
            enc_bits,
            pad_bits,
            pad_len,
            block_meta,
            params,
            intermediates,
        ) = concat.encode(in_bits)

        # Check intermediate ground truth
        assert "rs_encoded_bits" in intermediates
        assert "final_convolutional_bits" in intermediates
        rs_bits = intermediates["rs_encoded_bits"]
        assert len(rs_bits) == 64 * 8
        assert len(enc_bits) == (64 * 8 + 2) * 2  # RS bits + 2 tail bits * 2

        # Reverse decode
        decoded = concat.decode(
            encoded_bits=enc_bits,
            original_bit_length=300,
            intermediate_rs_length=len(rs_bits),
            block_boundaries=block_meta,
        )
        assert np.array_equal(decoded, in_bits)
