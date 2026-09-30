"""
test_64qam.py
Unit tests for 64-QAM demodulation and LLRs.
"""

import numpy as np
from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.utils import generate_synthetic_demod_test_data


def test_64qam_demodulation():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("64-QAM", num_symbols=200)
    engine = DemodulationEngine()
    res = engine.demodulate(sync_result={"modulation": "64-QAM", "modulation_family": "QAM"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)
