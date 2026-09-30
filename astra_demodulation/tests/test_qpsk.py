"""
test_qpsk.py
Unit tests for QPSK demodulation and LLRs.
"""

import numpy as np
from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.utils import generate_synthetic_demod_test_data


def test_qpsk_demodulation():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=200)
    engine = DemodulationEngine()
    res = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)
    assert len(res.phase_variants) == 4
