import math
import numpy as np
import torch

from astra_modulation_2d.src.preprocessing import IQPreprocessor
from astra_modulation_2d.src.spectrogram import SpectrogramGenerator


def test_1_stft_output_finite():
    """TEST 1: STFT output is completely finite with no NaN or Inf."""
    gen = SpectrogramGenerator(n_fft=128, hop_length=32)
    x = torch.randn(4, 2, 2048)
    spec = gen(x)
    assert torch.all(torch.isfinite(spec)), "Spectrogram output contains non-finite values."


def test_2_spectrogram_deterministic():
    """TEST 2: Spectrogram generation is 100% deterministic for identical inputs."""
    gen = SpectrogramGenerator(n_fft=128, hop_length=32)
    x = torch.randn(2, 2, 2048)
    s1 = gen(x)
    s2 = gen(x)
    assert torch.allclose(s1, s2, atol=1e-6), "Spectrogram generation is non-deterministic."


def test_3_spectrogram_dimensions_correct():
    """TEST 3: Spectrogram output dimensions match expected [B, 1, F, T]."""
    n_fft = 128
    hop = 32
    gen = SpectrogramGenerator(n_fft=n_fft, hop_length=hop)
    x = torch.randn(3, 2, 2048)
    spec = gen(x)
    # Expected F = n_fft = 128, T = 2048 // 32 + 1 = 65
    assert spec.dim() == 4
    assert spec.shape[0] == 3
    assert spec.shape[1] == 1
    assert spec.shape[2] == n_fft
    assert spec.shape[3] == 65


def test_4_fftshift_correct():
    """TEST 4: fftshift centers DC component at middle frequency bin."""
    gen_shifted = SpectrogramGenerator(n_fft=128, hop_length=32, fftshift=True, normalization="none")
    # Pure DC complex signal: constant real value 1.0 + 0j
    x_dc = torch.zeros(1, 2, 2048)
    x_dc[:, 0, :] = 1.0  # Constant DC in I-channel
    spec = gen_shifted(x_dc)  # [1, 1, 128, T]
    # Peak should occur at middle frequency bin (n_fft // 2 = 64)
    mid_bin = 128 // 2
    avg_spectrum = spec[0, 0].mean(dim=-1)
    peak_bin = int(torch.argmax(avg_spectrum))
    assert peak_bin == mid_bin, f"DC peak was at bin {peak_bin}, expected center bin {mid_bin}."


def test_5_dc_removal_correct():
    """TEST 5: Preprocessor successfully eliminates DC offset."""
    prep = IQPreprocessor(remove_dc=True, rms_normalization=False)
    # Signal with strong DC bias (+5.0)
    sig = np.random.randn(2048) + 1j * np.random.randn(2048) + (5.0 + 5.0j)
    clean = prep.process_numpy(sig)
    assert np.isclose(np.mean(clean), 0.0, atol=1e-6), "DC offset was not removed."


def test_6_rms_normalization_correct():
    """TEST 6: Preprocessor scales signal to unit RMS power (RMS = 1.0)."""
    prep = IQPreprocessor(remove_dc=True, rms_normalization=True)
    sig = (np.random.randn(2048) + 1j * np.random.randn(2048)) * 100.0
    clean = prep.process_numpy(sig)
    rms = np.sqrt(np.mean(np.abs(clean) ** 2))
    assert np.isclose(rms, 1.0, atol=1e-4), f"Normalized RMS power is {rms}, expected 1.0."


def test_19_nan_inf_handling():
    """TEST 19: Preprocessor safely replaces NaN/Inf without exploding."""
    prep = IQPreprocessor(strict_finite=True)
    sig = np.random.randn(2048) + 1j * np.random.randn(2048)
    sig[10] = np.nan + 1j * np.nan
    sig[50] = np.inf + 0j
    clean = prep.process_numpy(sig)
    assert np.all(np.isfinite(clean)), "NaN/Inf values leaked through preprocessor."
