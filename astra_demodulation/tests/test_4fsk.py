"""
test_4fsk.py
Unit tests for 4-FSK demodulation and Gray tone mapping.
"""

from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.utils import generate_synthetic_demod_test_data


def test_4fsk_demod():
    rx, _, tx_bits, _ = generate_synthetic_demod_test_data("4-FSK", num_symbols=150)
    engine = DemodulationEngine()
    res = engine.demodulate(sync_result={"modulation": "4-FSK", "modulation_family": "FSK"}, symbols=rx)
    assert res.success is True
    assert res.bit_count == 300
