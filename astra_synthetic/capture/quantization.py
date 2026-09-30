"""
Quantization, Scaling, and Clipping Modules for ASTRA Engine 7.
Performs linear integer mapping (int16, int8) with headroom protection and distortion metrics.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def apply_clipping(
    iq: np.ndarray,
    clip_level: float = 1.0,
) -> tuple[np.ndarray, int, float]:
    """Apply front-end amplitude clipping to complex baseband IQ.

    Clips the Cartesian I and Q components independently to [-clip_level, clip_level].

    Args:
        iq: 1D complex NumPy array.
        clip_level: Maximum allowable absolute amplitude per quadrature component.

    Returns:
        tuple (clipped_iq, clipped_sample_count, clipped_fraction)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), 0, 0.0

    real_clipped = np.clip(iq.real, -clip_level, clip_level)
    imag_clipped = np.clip(iq.imag, -clip_level, clip_level)

    # Count how many individual scalar I or Q values were clipped
    clipped_real_mask = np.abs(iq.real) > clip_level
    clipped_imag_mask = np.abs(iq.imag) > clip_level
    clipped_count = int(np.sum(clipped_real_mask) + np.sum(clipped_imag_mask))
    clipped_fraction = float(clipped_count / (2.0 * len(iq)))

    clipped_iq = (real_clipped + 1j * imag_clipped).astype(np.complex64)
    return clipped_iq, clipped_count, clipped_fraction


def quantize_scalars(
    scalars: np.ndarray,
    dtype_str: str = "int16",
    mode: str = "full_scale_peak",
    headroom_db: float = 1.0,
    fixed_scale: float | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Quantize a 1D real scalar stream into a signed integer format without overflow wrap.

    Supported datatypes: 'int16', 'int8', 'float32', 'float64'.

    Args:
        scalars: 1D real NumPy array (interleaved I/Q).
        dtype_str: Target storage data type ('int16', 'int8', 'float32', 'float64').
        mode: Quantization mode ('none', 'full_scale_peak', 'fixed_scale', 'target_rms').
        headroom_db: Headroom attenuation in dB (e.g. 1.0 dB prevents clipping on rounding).
        fixed_scale: Explicit scale multiplier if mode == 'fixed_scale'.

    Returns:
        tuple (quantized_array, scale_factor, mse_error, qsnr_db)
    """
    if len(scalars) == 0:
        target_np_dtype = np.dtype(dtype_str)
        return np.empty(0, dtype=target_np_dtype), 1.0, 0.0, float("inf")

    dtype_clean = dtype_str.lower().strip()

    if dtype_clean in ("float32", "float64", "f32", "f64"):
        target_dtype = np.float32 if "32" in dtype_clean else np.float64
        return scalars.astype(target_dtype), 1.0, 0.0, float("inf")

    # Determine integer boundaries
    if dtype_clean in ("int16", "i16"):
        min_val, max_val = -32768, 32767
        target_dtype = np.int16
    elif dtype_clean in ("int8", "i8"):
        min_val, max_val = -128, 127
        target_dtype = np.int8
    else:
        raise ValueError(f"Unsupported storage dtype: '{dtype_str}'")

    # Compute scale factor S
    peak_val = float(np.max(np.abs(scalars)))
    headroom_linear = 10.0 ** (-headroom_db / 20.0)

    if mode == "fixed_scale" and fixed_scale is not None:
        scale = float(fixed_scale)
    elif mode == "target_rms":
        rms_val = float(np.sqrt(np.mean(scalars**2)))
        target_int_rms = (max_val * headroom_linear) / 3.0
        scale = float(target_int_rms / max(1e-12, rms_val))
    else:  # full_scale_peak (default)
        if peak_val > 1e-12:
            scale = float((max_val * headroom_linear) / peak_val)
        else:
            scale = 1.0

    # Scale and saturate
    scaled = scalars * scale
    # Explicit clipping ensures no overflow wrap-around occurs
    clipped = np.clip(np.round(scaled), min_val, max_val)
    quantized = clipped.astype(target_dtype)

    # Compute Quantization Error & QSNR relative to input
    reconstructed = quantized.astype(np.float64) / scale
    err = scalars - reconstructed
    mse = float(np.mean(err**2))
    sig_power = float(np.mean(scalars**2))

    if mse > 1e-15 and sig_power > 1e-15:
        qsnr_db = float(10.0 * np.log10(sig_power / mse))
    else:
        qsnr_db = 100.0

    return quantized, scale, mse, qsnr_db


def dequantize_scalars(
    quantized: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Reconstruct continuous floating-point scalars from quantized integer array."""
    if len(quantized) == 0:
        return np.empty(0, dtype=np.float32)

    if scale <= 0:
        raise ValueError(f"Quantization scale must be > 0, got {scale}")

    return (quantized.astype(np.float32) / scale)
