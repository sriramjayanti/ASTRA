"""
test_bpsk.py
Unit tests for BPSK demodulation and LLRs.
"""

import numpy as np
from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.utils import generate_synthetic_demod_test_data


def test_bpsk_demodulation_and_llr():
    rx, tx_syms, tx_bits, _ = generate_synthetic_demod_test_data("BPSK", num_symbols=200)
    engine = DemodulationEngine()
    res = engine.demodulate(sync_result={"modulation": "BPSK", "modulation_family": "PSK"}, symbols=rx)
    assert res.success is True
    assert np.array_equal(res.hard_bits, tx_bits)
    assert len(res.soft_llrs) == len(tx_bits)
