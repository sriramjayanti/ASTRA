"""
Stereo WAV IQ File Writer and Reference Reader for ASTRA Engine 7.
Encapsulates 2-channel complex baseband recordings into standard RIFF WAV files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from .quantization import quantize_scalars, dequantize_scalars


def write_wav_iq_file(
    iq: np.ndarray,
    output_path: Path | str,
    sample_rate_hz: float,
    storage_dtype: str = "int16",
    iq_order: str = "IQ",
    quantization_mode: str = "full_scale_peak",
    headroom_db: float = 1.0,
    fixed_scale: float | None = None,
) -> tuple[Path, str, int, float, float, float]:
    """Write complex baseband IQ into a 2-channel WAV audio file.

    Channel 0 = I, Channel 1 = Q (or QI if configured).

    Args:
        iq: 1D complex NumPy array of baseband IQ samples.
        output_path: Target output WAV file path.
        sample_rate_hz: Sampling frequency in Hz.
        storage_dtype: 'int16' or 'float32'.
        iq_order: 'IQ' or 'QI'.
        quantization_mode: 'full_scale_peak', 'fixed_scale', 'target_rms', 'none'.
        headroom_db: Quantization headroom in dB.
        fixed_scale: Fixed scale factor if mode == 'fixed_scale'.

    Returns:
        tuple (written_path, file_sha256, file_size_bytes, scale, mse, qsnr_db)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate integer sample rate
    int_rate = int(round(sample_rate_hz))
    if int_rate <= 0:
        raise ValueError(f"WAV sample rate must be > 0, got {sample_rate_hz}")

    order = iq_order.upper().strip()
    if order not in ("IQ", "QI"):
        raise ValueError(f"Invalid iq_order: '{iq_order}'. Must be 'IQ' or 'QI'.")

    dtype_clean = storage_dtype.lower().strip()
    if dtype_clean not in ("int16", "i16", "float32", "f32"):
        raise ValueError(f"WAV IQ only supports 'int16' and 'float32', got '{storage_dtype}'")

    if len(iq) == 0:
        target_np_dtype = np.int16 if "int" in dtype_clean else np.float32
        wavfile.write(str(path), int_rate, np.empty((0, 2), dtype=target_np_dtype))
        with open(path, "rb") as f:
            raw_bytes = f.read()
        return path, hashlib.sha256(raw_bytes).hexdigest(), len(raw_bytes), 1.0, 0.0, float("inf")

    # Arrange 2D array (N, 2)
    if order == "IQ":
        ch0 = iq.real
        ch1 = iq.imag
    else:  # QI
        ch0 = iq.imag
        ch1 = iq.real

    # Stack into 1D scalar interleaved representation to quantize together with common scale
    scalars = np.empty(2 * len(iq), dtype=np.float32)
    scalars[0::2] = ch0
    scalars[1::2] = ch1

    if "int" in dtype_clean:
        quant_scalars, scale, mse, qsnr_db = quantize_scalars(
            scalars,
            dtype_str="int16",
            mode=quantization_mode,
            headroom_db=headroom_db,
            fixed_scale=fixed_scale,
        )
        wav_data = np.column_stack((quant_scalars[0::2], quant_scalars[1::2]))
    else:  # float32
        scale = 1.0
        mse = 0.0
        qsnr_db = float("inf")
        wav_data = np.column_stack((ch0, ch1)).astype(np.float32)

    # Write WAV file
    wavfile.write(str(path), int_rate, wav_data)

    with open(path, "rb") as f:
        file_bytes = f.read()

    file_size = len(file_bytes)
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()

    return path, file_sha256, file_size, scale, mse, qsnr_db


def read_wav_iq_file(
    file_path: Path | str,
    iq_order: str = "IQ",
    quantization_scale: float | None = None,
) -> tuple[np.ndarray, int]:
    """Reference reader for 2-channel WAV IQ files.

    Args:
        file_path: Path to the WAV file.
        iq_order: 'IQ' or 'QI'.
        quantization_scale: Optional scale factor for int16 WAV dequantization.

    Returns:
        tuple (iq_complex64, sample_rate_hz)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"WAV IQ file not found: {path}")

    rate, data = wavfile.read(str(path))

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected 2-channel stereo WAV, got shape {data.shape}")

    order = iq_order.upper().strip()
    ch0 = data[:, 0]
    ch1 = data[:, 1]

    if np.issubdtype(data.dtype, np.integer) and quantization_scale is not None:
        ch0 = dequantize_scalars(ch0, scale=quantization_scale)
        ch1 = dequantize_scalars(ch1, scale=quantization_scale)
    else:
        ch0 = ch0.astype(np.float32)
        ch1 = ch1.astype(np.float32)

    if order == "IQ":
        real = ch0
        imag = ch1
    else:  # QI
        imag = ch0
        real = ch1

    iq = (real + 1j * imag).astype(np.complex64)
    return iq, rate
