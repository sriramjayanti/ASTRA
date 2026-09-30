"""
test_16qam.py
Unit tests for 16-QAM demodulation and LLRs.
"""

import numpy as np
from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.utils import generate_synthetic_demod_test_data


def test_16qam_demodulation():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("16-QAM", num_symbols=200)
    engine = DemodulationEngine()
    res = engine.demodulate(sync_result={"modulation": "16-QAM", "modulation_family": "QAM"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)
    assert len(res.phase_variants) == 4
