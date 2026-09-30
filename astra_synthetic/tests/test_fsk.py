"""
Unit tests for FSK modulators (2-FSK, 4-FSK, MSK) in ASTRA Engine 5.
"""

import numpy as np
import pytest

from astra_synthetic.modulation.fsk import FSKModulator


class TestFSKModulator:
    """Test suite for Frequency Shift Keying modulations."""

    def test_2fsk_distinct_tones_and_roundtrip(self):
        """Verify 2-FSK continuous-phase synthesis and clean matched filter demodulation."""
        rng = np.random.default_rng(303)
        bits = rng.integers(0, 2, size=128, dtype=np.uint8)
        mod = FSKModulator(modulation_type="2fsk", continuous_phase=True, tone_spacing_ratio=1.0)

        clean_iq, ideal_syms, sym_idx, pad_bits, pad_len, mapped_len, params = mod.modulate(
            bits=bits, symbol_rate=9600.0, sample_rate=192000.0
        )

        sps = 20  # 192000 / 9600
        assert len(clean_iq) == 128 * sps
        assert clean_iq.dtype == np.complex64
        # Unit circle envelope for CPFSK
        np.testing.assert_allclose(np.abs(clean_iq), np.ones(len(clean_iq)), atol=1e-4)

        # Reference demodulation
        recovered = mod.reference_demodulate(
            clean_iq=clean_iq,
            symbol_rate=9600.0,
            sample_rate=192000.0,
            symbol_count=128,
            original_bit_length=len(bits),
        )
        np.testing.assert_array_equal(recovered, bits)

    def test_4fsk_four_tones_and_roundtrip(self):
        """Verify 4-FSK four distinct tones and clean demodulation."""
        rng = np.random.default_rng(404)
        bits = rng.integers(0, 2, size=200, dtype=np.uint8)  # 100 symbols
        mod = FSKModulator(modulation_type="4fsk", continuous_phase=True, tone_spacing_ratio=1.0)

        clean_iq, ideal_syms, sym_idx, pad_bits, pad_len, mapped_len, params = mod.modulate(
            bits=bits, symbol_rate=9600.0, sample_rate=192000.0
        )

        assert len(ideal_syms) == 100
        assert len(clean_iq) == 100 * 20

        recovered = mod.reference_demodulate(
            clean_iq=clean_iq,
            symbol_rate=9600.0,
            sample_rate=192000.0,
            symbol_count=100,
            original_bit_length=len(bits),
        )
        np.testing.assert_array_equal(recovered, bits)

    def test_non_integer_sps_raises_error(self):
        """Verify error is raised when sample_rate / symbol_rate is non-integer in Engine 5 v1."""
        mod = FSKModulator(modulation_type="2fsk")
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="Non-integer SPS"):
            mod.modulate(bits=bits, symbol_rate=10000.0, sample_rate=192000.0)
