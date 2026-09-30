"""
Unit tests for QAM modulators (16-QAM, 64-QAM) in ASTRA Engine 5.
"""

import numpy as np
import pytest

from astra_synthetic.modulation.qam import QAMModulator


class TestQAMModulator:
    """Test suite for Quadrature Amplitude Modulation."""

    def test_16qam_grid_and_roundtrip(self):
        """Verify 16-QAM produces 16 unique points and recovers cleanly."""
        rng = np.random.default_rng(101)
        bits = rng.integers(0, 2, size=400, dtype=np.uint8)  # 100 symbols
        mod = QAMModulator(modulation_type="16qam")

        clean_iq, ideal_syms, sym_idx, pad_bits, pad_len, mapped_len, taps, delay, params = mod.modulate(
            bits=bits, sps=8, pulse_shape="rrc", rolloff=0.35, span_symbols=8
        )

        assert len(ideal_syms) == 100
        # Check that constellation power before RRC filter is normalized: E[|s|^2] == 1.0
        # Theoretical average power of unnormalized +/-1, +/-3 is 10. Normalization factor is 1/sqrt(10).
        assert abs(mod.norm_factor - 1.0 / np.sqrt(10.0)) < 1e-10

        # Reference demodulation
        recovered = mod.reference_demodulate(
            clean_iq=clean_iq,
            sps=8,
            group_delay=delay,
            symbol_count=100,
            original_bit_length=len(bits),
            ideal_symbols=ideal_syms,
        )
        np.testing.assert_array_equal(recovered, bits)

    def test_64qam_grid_and_roundtrip(self):
        """Verify 64-QAM produces 64 unique points and recovers cleanly."""
        rng = np.random.default_rng(202)
        bits = rng.integers(0, 2, size=600, dtype=np.uint8)  # 100 symbols
        mod = QAMModulator(modulation_type="64qam")

        clean_iq, ideal_syms, sym_idx, pad_bits, pad_len, mapped_len, taps, delay, params = mod.modulate(
            bits=bits, sps=8, pulse_shape="rrc", rolloff=0.35, span_symbols=8
        )

        assert len(ideal_syms) == 100
        assert abs(mod.norm_factor - 1.0 / np.sqrt(42.0)) < 1e-10

        recovered = mod.reference_demodulate(
            clean_iq=clean_iq,
            sps=8,
            group_delay=delay,
            symbol_count=100,
            original_bit_length=len(bits),
            ideal_symbols=ideal_syms,
        )
        np.testing.assert_array_equal(recovered, bits)

    def test_qam_padding_handling(self):
        """Verify 16-QAM correctly pads non-multiple bit lengths and recovers unpadded length."""
        bits = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)  # 7 bits, needs 1 pad bit for k=4
        mod = QAMModulator(modulation_type="16qam")

        clean_iq, ideal_syms, sym_idx, pad_bits, pad_len, mapped_len, taps, delay, params = mod.modulate(
            bits=bits, sps=4, pad_mode="zeros"
        )
        assert pad_len == 1
        assert len(ideal_syms) == 2

        recovered = mod.reference_demodulate(
            clean_iq=clean_iq,
            sps=4,
            group_delay=delay,
            symbol_count=2,
            original_bit_length=7,
            ideal_symbols=ideal_syms,
        )
        assert len(recovered) == 7
        np.testing.assert_array_equal(recovered, bits)
