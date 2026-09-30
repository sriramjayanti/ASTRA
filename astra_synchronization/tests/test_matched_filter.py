"""
test_matched_filter.py
Unit tests for RRC matched filtering.
"""

import numpy as np
from astra_synchronization.src.matched_filter import design_rrc_filter, apply_matched_filter


def test_rrc_filter_properties():
    h = design_rrc_filter(sps=8.0, rolloff=0.25, span_symbols=6)
    assert len(h) > 0
    assert np.all(np.isfinite(h))
    assert np.allclose(h, h[::-1], atol=1e-5)
    
    x = np.ones(100, dtype=np.complex64)
    y, delay = apply_matched_filter(x, sps=8.0, rolloff=0.25, span_symbols=6)
    assert delay > 0
    assert len(y) == len(x)
