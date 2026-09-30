"""
test_autocorrelation.py
Unit tests for direct and FFT autocorrelation, peak detection, and harmonic analysis.
"""

import pytest
import numpy as np
from astra_bitstream_intelligence.src.autocorrelation import (
    bits_to_bipolar,
    direct_autocorrelation,
    fft_autocorrelation,
    bit_autocorrelation,
    find_autocorrelation_peaks
)


def test_5_autocorrelation_known_period():
    """TEST 5: autocorrelation identifies known period."""
    period = 64
    rng = np.random.RandomState(42)
    frame_template = rng.randint(0, 2, size=period, dtype=np.uint8)
    repeated = np.tile(frame_template, 16)

    corr = bit_autocorrelation(repeated, max_lag_fraction=0.5, max_lag_limit=512)
    peaks = find_autocorrelation_peaks(corr, min_lag=16, peak_threshold=0.5, top_k=5)

    assert len(peaks) > 0
    # Peak at lag 64 or multiple of 64
    lags = [p.lag for p in peaks]
    assert 64 in lags or any(abs(l - 64) <= 1 for l in lags)


def test_6_fft_and_direct_consistency():
    """TEST 6: FFT and direct autocorrelation produce consistent results."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, size=256, dtype=np.uint8)
    bipolar = bits_to_bipolar(bits)

    max_lag = 100
    r_direct = direct_autocorrelation(bipolar, max_lag=max_lag)
    r_fft = fft_autocorrelation(bipolar, max_lag=max_lag)

    assert np.allclose(r_direct, r_fft, atol=1e-3)


def test_7_peak_detection():
    """TEST 7: peak detection excludes lag 0 and identifies true peaks."""
    # Synthetic correlation curve with known peak at lag 128
    corr = np.zeros(256, dtype=np.float32)
    corr[0] = 1.0
    corr[128] = 0.85
    corr[127] = 0.50
    corr[129] = 0.50

    peaks = find_autocorrelation_peaks(corr, min_lag=16, peak_threshold=0.2)
    assert len(peaks) == 1
    assert peaks[0].lag == 128
    assert abs(peaks[0].correlation - 0.85) < 1e-3


def test_26_large_stream_fft_path():
    """TEST 26: large bitstream automatically uses FFT path."""
    rng = np.random.RandomState(42)
    large_bits = rng.randint(0, 2, size=8192, dtype=np.uint8)
    corr = bit_autocorrelation(large_bits, fft_threshold=2048)
    assert len(corr) > 100
    assert corr[0] == pytest.approx(1.0, abs=1e-3)
