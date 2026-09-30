"""
test_llr.py
Unit tests for Log-Likelihood Ratio calculation.
"""

import numpy as np
from astra_demodulation.src.mappings import get_constellation
from astra_demodulation.src.llr import compute_soft_llrs


def test_llr_sign_and_exact_vs_maxlog():
    const = get_constellation("QPSK")
    soft_exact = compute_soft_llrs(const.complex_points, const, noise_variance=0.05, mode="exact")
    soft_maxlog = compute_soft_llrs(const.complex_points, const, noise_variance=0.05, mode="max_log")
    
    assert np.all(np.sign(soft_exact.llrs) == np.sign(soft_maxlog.llrs))
    assert len(soft_exact.llrs) == 8
