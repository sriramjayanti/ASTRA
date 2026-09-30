"""
test_costas.py
Unit tests for Costas loop carrier recovery.
"""

import numpy as np
from astra_synchronization.src.costas import costas_recover, get_phase_ambiguity_states
from astra_synchronization.src.utils import generate_synthetic_symbols


def test_qpsk_costas_loop():
    symbols = generate_synthetic_symbols("QPSK", num_symbols=400)
    # Apply phase rotation
    phase_rot = 0.5
    rot_syms = symbols * np.exp(1j * phase_rot)
    
    corr_syms, est_phase, lock, state, amb = costas_recover(rot_syms, modulation="QPSK")
    assert lock > 0.45
    assert len(amb) == 4
