"""
test_ambiguity.py
Unit tests for phase ambiguity resolution and candidate bitstream generation.
"""

import numpy as np
from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.utils import generate_synthetic_demod_test_data


def test_qpsk_rotations():
    rx, _, tx_bits, _ = generate_synthetic_demod_test_data("QPSK", num_symbols=100)
    engine = DemodulationEngine()
    # Test 90, 180, 270 deg rotations
    for deg in [90.0, 180.0, 270.0]:
        rot_rx = rx * np.exp(1j * np.deg2rad(deg))
        res = engine.demodulate(sync_result={"modulation": "QPSK", "modulation_family": "PSK"}, symbols=rot_rx)
        var = next(v for v in res.phase_variants if abs(v.rotation_deg - deg) < 1.0)
        assert np.array_equal(var.hard_bits, tx_bits)
