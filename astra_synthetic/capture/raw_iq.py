"""
Raw Binary IQ File Writer and Reference Reader for ASTRA Engine 7.
Handles writing and parsing raw interleaved binary .iq and .bin SDR recordings with explicit endianness.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from .layouts import format_iq_scalar_stream, deformat_iq_scalar_stream
from .quantization import quantize_scalars, dequantize_scalars


def resolve_numpy_dtype(dtype_str: str, endianness: str = "little") -> np.dtype:
    """Resolve storage dtype and byte-order into a NumPy dtype object.

    Args:
        dtype_str: 'float32', 'float64', 'int16', 'int8'.
        endianness: 'little' ('<') or 'big' ('>').

    Returns:
        np.dtype object.
    """
    clean_dt = dtype_str.lower().strip()
    endian_prefix = "<" if endianness.lower() in ("little", "le", "<") else ">"

    if clean_dt in ("float32", "f32"):
        return np.dtype(f"{endian_prefix}f4")
    elif clean_dt in ("float64", "f64"):
        return np.dtype(f"{endian_prefix}f8")
    elif clean_dt in ("int16", "i16"):
        return np.dtype(f"{endian_prefix}i2")
    elif clean_dt in ("int8", "i8"):
        return np.dtype("i1")  # Single-byte has no endianness
    else:
        raise ValueError(f"Unsupported dtype: '{dtype_str}'")


def write_raw_iq_file(
    iq: np.ndarray,
    output_path: Path | str,
    storage_dtype: str = "float32",
    endianness: str = "little",
    iq_order: str = "IQ",
    quantization_mode: str = "full_scale_peak",
    headroom_db: float = 1.0,
    fixed_scale: float | None = None,
) -> tuple[Path, str, int, float, float, float]:
    """Write complex baseband IQ to a raw binary .iq file.

    Args:
        iq: 1D complex NumPy array of baseband IQ samples.
        output_path: Target output file path.
        storage_dtype: 'float32', 'int16', 'int8', 'float64'.
        endianness: 'little' or 'big'.
        iq_order: 'IQ' or 'QI'.
        quantization_mode: 'none', 'full_scale_peak', 'fixed_scale', 'target_rms'.
        headroom_db: Quantization headroom in dB.
        fixed_scale: Fixed quantization scale multiplier if mode == 'fixed_scale'.

    Returns:
        tuple (written_path, file_sha256, file_size_bytes, scale, mse, qsnr_db)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Format into 1D real scalar stream
    scalars = format_iq_scalar_stream(iq, iq_order=iq_order)

    # 2. Quantize scalars if integer dtype
    quant_data, scale, mse, qsnr_db = quantize_scalars(
        scalars,
        dtype_str=storage_dtype,
        mode=quantization_mode if "int" in storage_dtype.lower() else "none",
        headroom_db=headroom_db,
        fixed_scale=fixed_scale,
    )

    # 3. Apply target byte-order
    np_dtype = resolve_numpy_dtype(storage_dtype, endianness=endianness)
    final_array = quant_data.astype(np_dtype)

    # 4. Write binary bytes
    raw_bytes = final_array.tobytes()
    with open(path, "wb") as f:
        f.write(raw_bytes)

    file_size = len(raw_bytes)
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    return path, file_sha256, file_size, scale, mse, qsnr_db


def read_raw_iq_file(
    file_path: Path | str,
    storage_dtype: str = "float32",
    endianness: str = "little",
    iq_order: str = "IQ",
    quantization_scale: float | None = None,
) -> np.ndarray:
    """Reference reader for raw binary .iq files. Reconstructs complex64 baseband IQ.

    Args:
        file_path: Path to the binary file.
        storage_dtype: 'float32', 'int16', 'int8', 'float64'.
        endianness: 'little' or 'big'.
        iq_order: 'IQ' or 'QI'.
        quantization_scale: Scale factor to dequantize integer data back to floating-point.

    Returns:
        1D complex64 NumPy array.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw IQ file not found: {path}")

    np_dtype = resolve_numpy_dtype(storage_dtype, endianness=endianness)
    raw_data = np.fromfile(path, dtype=np_dtype)

    # Dequantize if integer dtype
    if "int" in storage_dtype.lower() and quantization_scale is not None:
        scalars = dequantize_scalars(raw_data, scale=quantization_scale)
    else:
        scalars = raw_data.astype(np.float32)

    # De-interleave to complex64
    return deformat_iq_scalar_stream(scalars, iq_order=iq_order)
