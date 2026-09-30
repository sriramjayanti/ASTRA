"""
Unit tests for integer Quantization and Clipping modules (Engine 7).
"""

import numpy as np
import pytest
from astra_synthetic.capture.quantization import apply_clipping, quantize_scalars, dequantize_scalars


def test_clipping_saturation_and_counter():
    # Signal with peaks exceeding 1.0
    x = np.array([0.5 + 0.5j, 1.5 - 2.0j, -0.8 + 0.0j], dtype=np.complex64)
    clipped_iq, count, frac = apply_clipping(x, clip_level=1.0)

    # 1.5 was clipped to 1.0, -2.0 was clipped to -1.0 (2 clips total)
    assert count == 2
    assert np.isclose(frac, 2.0 / 6.0)
    assert np.all(np.abs(clipped_iq.real) <= 1.0)
    assert np.all(np.abs(clipped_iq.imag) <= 1.0)


def test_quantization_no_overflow_wrap():
    # If values exceed nominal scale, clipping prevents modulo wrap-around
    scalars = np.array([1000.0, -1000.0], dtype=np.float32)
    # Using fixed scale that forces max boundary
    quant_i16, scale, _, _ = quantize_scalars(scalars, dtype_str="int16", mode="fixed_scale", fixed_scale=100.0)

    # 1000 * 100 = 100,000 > 32767 -> must be clamped to 32767, NOT wrapped to negative!
    assert quant_i16[0] == 32767
    assert quant_i16[1] == -32768


def test_int16_quantization_snr():
    rng = np.random.default_rng(123)
    scalars = rng.standard_normal(10000).astype(np.float32)

    quant_i16, scale, mse, qsnr_db = quantize_scalars(scalars, dtype_str="int16", mode="full_scale_peak")

    # 16-bit theoretical quantization SNR is ~98 dB (6.02*16 + 1.76 - peak factor)
    assert qsnr_db > 70.0
    assert mse < 1e-4

    # Dequantized array must match original within small MSE
    reconstructed = dequantize_scalars(quant_i16, scale)
    assert np.allclose(scalars, reconstructed, atol=1e-3)


def test_int8_quantization_snr():
    rng = np.random.default_rng(123)
    scalars = rng.standard_normal(10000).astype(np.float32)

    quant_i8, scale, mse, qsnr_db = quantize_scalars(scalars, dtype_str="int8", mode="full_scale_peak")

    # 8-bit quantization SNR is typically ~30-45 dB
    assert 25.0 < qsnr_db < 55.0
    reconstructed = dequantize_scalars(quant_i8, scale)
    assert np.allclose(scalars, reconstructed, atol=0.1)
