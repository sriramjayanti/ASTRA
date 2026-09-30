"""
Unit tests for PSK modulators (BPSK, QPSK, 8PSK) in ASTRA Engine 5.
"""

import numpy as np
import pytest

from astra_synthetic.modulation.psk import PSKModulator


class TestPSKModulator:
    """Test suite for Phase Shift Keying modulations."""

    def test_bpsk_known_mapping_and_roundtrip(self):
        """Verify BPSK maps bit 0 -> +1, bit 1 -> -1 and recovers exactly."""
        bits = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
        mod = PSKModulator(modulation_type="bpsk")

        clean_iq, ideal_syms, sym_idx, pad_bits, pad_len, mapped_len, taps, delay, params = mod.modulate(
            bits=bits, sps=8, pulse_shape="rrc", rolloff=0.35, span_symbols=8
        )

        # Ideal symbols must be +1, -1
        expected_syms = np.array([1.0 + 0j, -1.0 + 0j, -1.0 + 0j, 1.0 + 0j, -1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
        np.testing.assert_allclose(ideal_syms, expected_syms, atol=1e-6)

        # Average power should be normalized to ~1.0
        avg_pwr = np.mean(np.abs(clean_iq) ** 2)
        assert abs(avg_pwr - 1.0) < 0.05

        # Reference demodulation
        recovered_bits = mod.reference_demodulate(
            clean_iq=clean_iq,
            sps=8,
            group_delay=delay,
            symbol_count=len(ideal_syms),
            original_bit_length=len(bits),
            ideal_symbols=ideal_syms,
        )
        np.testing.assert_array_equal(recovered_bits, bits)

    def test_qpsk_all_four_points_and_gray_roundtrip(self):
        """Verify all 4 QPSK bit combinations map to 4 distinct unit-magnitude points."""
        # 00, 01, 11, 10
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        mod = PSKModulator(modulation_type="qpsk")

        clean_iq, ideal_syms, sym_idx, pad_bits, pad_len, mapped_len, taps, delay, params = mod.modulate(
            bits=bits, sps=16, pulse_shape="rrc", rolloff=0.35, span_symbols=8
        )

        assert len(ideal_syms) == 4
        # Points must lie on unit circle |s| = 1.0
        np.testing.assert_allclose(np.abs(ideal_syms), np.ones(4), atol=1e-5)

        # All 4 ideal symbols must be distinct
        assert len(np.unique(ideal_syms.round(decimals=3))) == 4

        # Roundtrip recovery
        recovered = mod.reference_demodulate(
            clean_iq=clean_iq,
            sps=16,
            group_delay=delay,
            symbol_count=4,
            original_bit_length=8,
            ideal_symbols=ideal_syms,
        )
        np.testing.assert_array_equal(recovered, bits)

    def test_8psk_equidistant_phases_and_roundtrip(self):
        """Verify 8-PSK generates 8 distinct phase points on unit circle and roundtrips cleanly."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=300, dtype=np.uint8)  # 100 symbols
        mod = PSKModulator(modulation_type="8psk")

        clean_iq, ideal_syms, sym_idx, pad_bits, pad_len, mapped_len, taps, delay, params = mod.modulate(
            bits=bits, sps=8, pulse_shape="rrc", rolloff=0.35, span_symbols=8
        )

        assert len(ideal_syms) == 100
        np.testing.assert_allclose(np.abs(ideal_syms), np.ones(100), atol=1e-5)

        recovered = mod.reference_demodulate(
            clean_iq=clean_iq,
            sps=8,
            group_delay=delay,
            symbol_count=100,
            original_bit_length=len(bits),
            ideal_symbols=ideal_syms,
        )
        np.testing.assert_array_equal(recovered, bits)
