"""
Unit tests for Root Raised Cosine (RRC) filter and pulse shaping in ASTRA Engine 5.
"""

import numpy as np
import pytest

from astra_synthetic.modulation.filters import apply_rrc_pulse_shaping, root_raised_cosine_filter
from astra_synthetic.modulation.pulse_shaping import compute_evm_rms, extract_matched_symbols


class TestRRCFilter:
    """Test suite for analytical RRC filter design."""

    def test_rrc_filter_symmetry(self):
        """Verify RRC filter impulse response is strictly symmetric: h[n] == h[N-1-n]."""
        for beta in [0.0, 0.2, 0.35, 0.5, 1.0]:
            h = root_raised_cosine_filter(beta=beta, span_symbols=8, sps=8)
            np.testing.assert_allclose(h, h[::-1], atol=1e-12)

    def test_rrc_filter_unit_energy(self):
        """Verify RRC filter is normalized to unit energy: sum(h^2) == 1.0."""
        for beta in [0.2, 0.25, 0.35, 0.5]:
            h = root_raised_cosine_filter(beta=beta, span_symbols=10, sps=4)
            energy = np.sum(h ** 2)
            assert abs(energy - 1.0) < 1e-10

    def test_rrc_singular_point_evaluations(self):
        """Verify singularities at t=0 and t=+-1/(4*beta) are computed without NaN or Inf."""
        # For beta=0.25, sps=4, singularity is at t = +-1/(4*0.25) = +-1.0 symbol (which lands exactly on a sample)
        h = root_raised_cosine_filter(beta=0.25, span_symbols=6, sps=4)
        assert np.all(np.isfinite(h))
        assert not np.any(np.isnan(h))
        # Center tap (t=0) should be the maximum tap
        center_idx = len(h) // 2
        assert np.argmax(h) == center_idx

    def test_rrc_pulse_shaping_upsampling_and_group_delay(self):
        """Verify upsampling length and filter group delay."""
        symbols = np.array([1.0 + 0j, -1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
        sps = 8
        span = 6
        clean_iq, taps, group_delay = apply_rrc_pulse_shaping(symbols, sps=sps, beta=0.35, span_symbols=span)

        assert group_delay == (span * sps) // 2
        # Expected length = (num_symbols - 1) * sps + num_taps = 3 * 8 + (6*8 + 1) = 24 + 49 = 73 (or len(symbols)*sps + span*sps)
        assert len(clean_iq) >= len(symbols) * sps
        assert clean_iq.dtype == np.complex64

    def test_evm_rms_calculation(self):
        """Verify EVM is 0.0% for identical symbols and positive for perturbed symbols."""
        ref = np.array([1.0 + 1j, -1.0 + 1j, -1.0 - 1j, 1.0 - 1j], dtype=np.complex64)
        assert compute_evm_rms(ref, ref) == 0.0

        # Perturb with small delta
        est = ref + 0.01 * (1.0 + 1j)
        evm = compute_evm_rms(est, ref)
        assert 0.5 < evm < 2.0
