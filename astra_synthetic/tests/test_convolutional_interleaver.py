"""
Unit tests for Convolutional Interleaver and Deinterleaver in ASTRA Synthetic Engine 4.
Validates multi-branch delay structures, deterministic streaming output,
flush tail handling, and bit-exact reference deinterleaving round-trip.
"""

import numpy as np
import pytest

from astra_synthetic.interleaving.convolutional import (
    ConvolutionalInterleaver,
    deinterleave_convolutional,
    interleave_convolutional,
)


class TestConvolutionalInterleaver:
    """Test suite for Ramsey/Forney convolutional interleaving."""

    def test_branch_delay_initialization(self):
        """Verify branch delays are computed correctly: delay_i = i * M."""
        conv = ConvolutionalInterleaver(num_branches=4, delay_step=2)
        assert conv.branch_delays == [0, 2, 4, 6]
        assert conv.flush_length == (4 - 1) * 2 * 4  # (B-1)*M*B = 3*2*4 = 24 bits

        conv8 = ConvolutionalInterleaver(num_branches=8, delay_step=4)
        assert conv8.branch_delays == [0, 4, 8, 12, 16, 20, 24, 28]
        assert conv8.flush_length == 7 * 4 * 8  # 224 bits

    def test_deterministic_small_output(self):
        """Verify deterministic bit output through shift registers."""
        # 4 branches, step = 1 -> delays [0, 1, 2, 3]
        # flush length = 3 * 1 * 4 = 12 bits
        inp = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        interleaved, flush_bits, meta = interleave_convolutional(inp, num_branches=4, delay_step=1)

        # Total output length = len(inp) + flush_length = 8 + 12 = 20 bits
        assert len(interleaved) == 20
        assert len(flush_bits) == 12
        assert meta["latency_bits"] == 12

        # Roundtrip recovery
        recovered = deinterleave_convolutional(
            interleaved, num_branches=4, delay_step=1, original_bit_length=len(inp)
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_clean_round_trip_b4_d2(self):
        """Test round-trip reconstruction for conv_b4_d2 profile."""
        rng = np.random.default_rng(101)
        inp = rng.integers(0, 2, size=500, dtype=np.uint8)

        interleaved, flush_bits, meta = interleave_convolutional(inp, num_branches=4, delay_step=2)
        assert len(interleaved) == 500 + meta["flush_length"]

        recovered = deinterleave_convolutional(
            interleaved, num_branches=4, delay_step=2, original_bit_length=500
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_clean_round_trip_b8_d4(self):
        """Test round-trip reconstruction for conv_b8_d4 profile with larger latency."""
        rng = np.random.default_rng(202)
        inp = rng.integers(0, 2, size=1500, dtype=np.uint8)

        interleaved, flush_bits, meta = interleave_convolutional(inp, num_branches=8, delay_step=4)
        recovered = deinterleave_convolutional(
            interleaved, num_branches=8, delay_step=4, original_bit_length=1500
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_non_multiple_length_input(self):
        """Test interleaver when input length is not a multiple of num_branches."""
        rng = np.random.default_rng(303)
        inp = rng.integers(0, 2, size=137, dtype=np.uint8)

        interleaved, flush_bits, meta = interleave_convolutional(inp, num_branches=4, delay_step=3)
        recovered = deinterleave_convolutional(
            interleaved, num_branches=4, delay_step=3, original_bit_length=137
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_invalid_parameters(self):
        """Verify errors on invalid branch configurations."""
        with pytest.raises(ValueError):
            ConvolutionalInterleaver(num_branches=1, delay_step=2)
        with pytest.raises(ValueError):
            ConvolutionalInterleaver(num_branches=4, delay_step=0)
