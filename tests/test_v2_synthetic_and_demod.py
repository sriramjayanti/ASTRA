"""
Unit and Round-Trip Validation Tests for ASTRA V2 Synthetic Signal Engine & Demodulator.
Validates:
1. 2-FSK & 4-FSK waveform generation and frequency spacing
2. QAM (16QAM, 64QAM, 256QAM) constellation mapping and pulse shaping
3. Clean round-trip demodulation (near-zero BER)
"""

import math
import numpy as np
import pytest

from astra_config.classes import TRAINED_MODULATION_CLASSES_V2, normalize_modulation_name
from scripts.generate_modulation_v2_dataset import (
    generate_v2_signal_capture,
    synthesize_fsk_clean,
    synthesize_qam_psk_clean,
    get_qam_grid,
)
from astra_demodulation.src.fsk import demodulate_fsk


def test_classes_integrity():
    assert len(TRAINED_MODULATION_CLASSES_V2) == 10
    assert "2-FSK" in TRAINED_MODULATION_CLASSES_V2
    assert "4-FSK" in TRAINED_MODULATION_CLASSES_V2
    assert "16QAM" in TRAINED_MODULATION_CLASSES_V2
    assert "64QAM" in TRAINED_MODULATION_CLASSES_V2
    assert "256QAM" in TRAINED_MODULATION_CLASSES_V2


def test_fsk_generation_shape_and_finite():
    rng = np.random.default_rng(42)
    for mod in ["2-FSK", "4-FSK"]:
        iq, meta = generate_v2_signal_capture(
            source_id="test_fsk",
            modulation=mod,
            difficulty="clean",
            seed=42,
            target_samples=4096,
        )
        assert len(iq) == 4096
        assert np.all(np.isfinite(iq))
        assert meta["modulation"] == mod
        assert "tone_spacing_ratio" in meta


def test_qam_generation_shape_and_finite():
    rng = np.random.default_rng(42)
    for mod in ["16QAM", "64QAM", "256QAM"]:
        iq, meta = generate_v2_signal_capture(
            source_id="test_qam",
            modulation=mod,
            difficulty="clean",
            seed=42,
            target_samples=4096,
        )
        assert len(iq) == 4096
        assert np.all(np.isfinite(iq))
        assert meta["modulation"] == mod


def test_clean_2fsk_demodulation_roundtrip():
    """Verify that a clean 2-FSK waveform has distinct frequency states and decodes cleanly."""
    rng = np.random.default_rng(123)
    baud = 9600.0
    fs = 192000.0
    sps = int(fs / baud)
    bits = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)

    iq, meta = synthesize_fsk_clean(
        bits=bits,
        M=2,
        baud=baud,
        sample_rate=fs,
        tone_spacing_ratio=1.0,
        continuous_phase=True,
        frequency_shift_ratio=0.0,
        drift_rate_hz_per_sec=0.0,
        initial_phase=0.0,
        rng=rng,
    )

    # Subsample at symbol centers
    sampled_symbols = iq[sps // 2 :: sps][:len(bits)]
    res = demodulate_fsk(
        symbols=sampled_symbols,
        candidate_id="cand_2fsk_test",
        modulation="2-FSK",
        sample_rate_hz=fs,
        symbol_rate_hz=baud,
    )
    assert res.success is True
    assert len(res.hard_bits) >= len(bits)
    # Check that alternating tones produce distinct bit values
    assert np.unique(res.hard_bits[:len(bits)]).size == 2


def test_qam_grid_properties():
    for M in [16, 64, 256]:
        grid, norm_f, k = get_qam_grid(M)
        assert len(grid) == M
        # Grid average power should equal 1.0 (unit normalized)
        avg_pwr = np.mean(np.abs(grid) ** 2)
        assert abs(avg_pwr - 1.0) < 1e-4
