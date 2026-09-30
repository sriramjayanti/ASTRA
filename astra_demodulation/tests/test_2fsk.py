"""
test_2fsk.py
Unit tests for 2-FSK demodulation and tone swap variants.
"""

from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.utils import generate_synthetic_demod_test_data


def test_2fsk_demod():
    rx, _, tx_bits, _ = generate_synthetic_demod_test_data("2-FSK", num_symbols=150)
    engine = DemodulationEngine()
    res = engine.demodulate(sync_result={"modulation": "2-FSK", "modulation_family": "FSK"}, symbols=rx)
    assert res.success is True
    assert res.bit_count == 150
    assert len(res.phase_variants) == 2
