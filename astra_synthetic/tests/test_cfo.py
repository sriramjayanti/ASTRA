"""
Unit tests for Carrier Frequency Offset (CFO) generator (Engine 6).
"""

import numpy as np
import pytest
from astra_synthetic.channel.cfo import apply_cfo


def test_cfo_zero():
    x = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64)
    y = apply_cfo(x, cfo_hz=0.0, sample_rate=100000.0)
    assert np.allclose(x, y)
    assert y.dtype == np.complex64


def test_cfo_spectral_shift():
    Fs = 100000.0
    N = 4096
    t = np.arange(N) / Fs
    # Baseband carrier at 1 kHz
    f_in = 1000.0
    x = np.exp(1j * 2 * np.pi * f_in * t).astype(np.complex64)

    # Apply 2.5 kHz CFO -> Resulting tone should be at 3.5 kHz
    cfo_hz = 2500.0
    y = apply_cfo(x, cfo_hz=cfo_hz, sample_rate=Fs)

    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, 1 / Fs)
    peak_freq = freqs[np.argmax(np.abs(fft_y))]

    assert np.isclose(peak_freq, f_in + cfo_hz, atol=Fs / N)


def test_cfo_negative_frequency():
    Fs = 50000.0
    N = 2048
    t = np.arange(N) / Fs
    f_in = 5000.0
    x = np.exp(1j * 2 * np.pi * f_in * t).astype(np.complex64)

    cfo_hz = -2000.0
    y = apply_cfo(x, cfo_hz=cfo_hz, sample_rate=Fs)

    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, 1 / Fs)
    peak_freq = freqs[np.argmax(np.abs(fft_y))]

    assert np.isclose(peak_freq, f_in + cfo_hz, atol=Fs / N)
