"""
Unit tests for ASTRA Convolutional Encoder and Viterbi Reference Decoder.
Validates known test vectors, zero-tail termination, clean round-trip, and error correction.
"""

import numpy as np
import pytest

from astra_synthetic.fec.convolutional import ConvolutionalCode, octal_to_taps


class TestConvolutionalCode:
    """Test shift-register encoding and Viterbi decoding."""

    def test_octal_to_taps_conversion(self):
        # 0o7 in K=3 is 111 binary
        assert octal_to_taps(0o7, constraint_length=3) == [1, 1, 1]
        # 0o5 in K=3 is 101 binary
        assert octal_to_taps(0o5, constraint_length=3) == [1, 0, 1]
        # 0o171 in K=7 is 1111001 binary
        assert octal_to_taps(0o171, constraint_length=7) == [1, 1, 1, 1, 0, 0, 1]
        # 0o133 in K=7 is 1011011 binary
        assert octal_to_taps(0o133, constraint_length=7) == [1, 0, 1, 1, 0, 1, 1]

    def test_known_vector_k3_r12(self):
        """Test K=3 Rate 1/2 with generators (7, 5) octal on a short input vector."""
        conv = ConvolutionalCode(
            constraint_length=3,
            generators=[0o7, 0o5],
            rate="1/2",
            termination_mode="zero_tail",
        )
        # Input: [1, 0, 1, 1] + 2 tail zeros = [1, 0, 1, 1, 0, 0]
        # t=0: in=1, reg=[1, 0, 0] -> out=(1^0^0, 1^0^0) = (1, 1)
        # t=1: in=0, reg=[0, 1, 0] -> out=(0^1^0, 0^0^0) = (1, 0)
        # t=2: in=1, reg=[1, 0, 1] -> out=(1^0^1, 1^0^1) = (0, 0)
        # t=3: in=1, reg=[1, 1, 0] -> out=(1^1^0, 1^0^0) = (0, 1)
        # t=4 (tail): in=0, reg=[0, 1, 1] -> out=(0^1^1, 0^0^1) = (0, 1)
        # t=5 (tail): in=0, reg=[0, 0, 1] -> out=(0^0^1, 0^0^1) = (1, 1)
        in_bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        encoded, tail_bits, tail_len, params = conv.encode(in_bits, termination="zero_tail")

        expected_enc = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
        assert len(encoded) == (4 + 2) * 2 == 12
        assert np.array_equal(encoded, expected_enc)
        assert tail_len == 2
        assert np.array_equal(tail_bits, np.array([0, 0], dtype=np.uint8))

    def test_clean_viterbi_round_trip_k7(self):
        """Test clean round-trip decoding for K=7 Rate 1/2 NASA code."""
        conv = ConvolutionalCode(
            constraint_length=7,
            generators=[0o171, 0o133],
            rate="1/2",
            termination_mode="zero_tail",
        )
        rng = np.random.default_rng(42)
        in_bits = rng.integers(0, 2, size=512, dtype=np.uint8)

        encoded, _, _, _ = conv.encode(in_bits)
        assert len(encoded) == (512 + 6) * 2

        decoded = conv.decode_viterbi(encoded, original_bit_length=512)
        assert np.array_equal(decoded, in_bits)

    def test_viterbi_error_correction(self):
        """Test that Viterbi decoder corrects isolated channel bit flips."""
        conv = ConvolutionalCode(
            constraint_length=7,
            generators=[0o171, 0o133],
            rate="1/2",
            termination_mode="zero_tail",
        )
        rng = np.random.default_rng(123)
        in_bits = rng.integers(0, 2, size=256, dtype=np.uint8)

        encoded, _, _, _ = conv.encode(in_bits)
        
        # Corrupt 3 well-spaced bits in the codeword
        corrupted = encoded.copy()
        corrupted[10] ^= 1
        corrupted[100] ^= 1
        corrupted[200] ^= 1

        decoded = conv.decode_viterbi(corrupted, original_bit_length=256)
        assert np.array_equal(decoded, in_bits)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="Constraint length K must be >= 2"):
            ConvolutionalCode(constraint_length=1)
