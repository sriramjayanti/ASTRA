"""
SigMF (Signal Metadata Format) Writer and Reference Reader for ASTRA Engine 7.
Generates compliant paired .sigmf-data and .sigmf-meta SDR datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from .raw_iq import write_raw_iq_file, read_raw_iq_file


SIGMF_DTYPE_MAP = {
    ("float32", "little"): "cf32_le",
    ("float32", "big"): "cf32_be",
    ("float64", "little"): "cf64_le",
    ("float64", "big"): "cf64_be",
    ("int16", "little"): "ci16_le",
    ("int16", "big"): "ci16_be",
    ("int8", "little"): "ci8",
    ("int8", "big"): "ci8",
}

REVERSE_SIGMF_MAP = {v: k for k, v in SIGMF_DTYPE_MAP.items()}


def write_sigmf_dataset(
    iq: np.ndarray,
    output_base_path: Path | str,
    sample_rate_hz: float,
    center_frequency_hz: float | None = None,
    storage_dtype: str = "float32",
    endianness: str = "little",
    quantization_mode: str = "full_scale_peak",
    headroom_db: float = 1.0,
    author: str = "ASTRA Synthetic Engine",
    description: str = "ASTRA Baseband RF Capture",
) -> tuple[Path, Path, str, int, float, float, float]:
    """Write SigMF-compliant dataset (.sigmf-data binary and .sigmf-meta JSON).

    Args:
        iq: 1D complex NumPy array of baseband IQ samples.
        output_base_path: Target path (without extension or with .sigmf-data).
        sample_rate_hz: Sample rate in Hz.
        center_frequency_hz: Optional RF center frequency in Hz.
        storage_dtype: 'float32', 'int16', 'int8'.
        endianness: 'little' or 'big'.
        quantization_mode: Quantization mode for integer storage.
        headroom_db: Quantization headroom.
        author: SigMF author metadata.
        description: SigMF dataset description.

    Returns:
        tuple (data_path, meta_path, data_file_sha256, file_size_bytes, scale, mse, qsnr_db)
    """
    base = Path(output_base_path)
    if base.name.endswith(".sigmf-data") or base.name.endswith(".sigmf-meta"):
        base = base.with_suffix("")

    data_path = base.parent / f"{base.name}.sigmf-data"
    meta_path = base.parent / f"{base.name}.sigmf-meta"

    # Write .sigmf-data binary using raw IQ engine (SigMF standard is always interleaved IQ)
    written_data_path, file_sha256, file_size, scale, mse, qsnr_db = write_raw_iq_file(
        iq=iq,
        output_path=data_path,
        storage_dtype=storage_dtype,
        endianness=endianness,
        iq_order="IQ",
        quantization_mode=quantization_mode,
        headroom_db=headroom_db,
    )

    # Determine SigMF datatype string
    sigmf_dtype = SIGMF_DTYPE_MAP.get(
        (storage_dtype.lower(), endianness.lower()), "cf32_le"
    )

    capture_segment: dict[str, Any] = {
        "core:sample_start": 0,
    }
    if center_frequency_hz is not None:
        capture_segment["core:frequency"] = float(center_frequency_hz)

    sigmf_meta: dict[str, Any] = {
        "global": {
            "core:datatype": sigmf_dtype,
            "core:sample_rate": float(sample_rate_hz),
            "core:version": "1.0.0",
            "core:description": description,
            "core:author": author,
            "core:sha512": None,
        },
        "captures": [capture_segment],
        "annotations": [],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sigmf_meta, f, indent=2)

    return data_path, meta_path, file_sha256, file_size, scale, mse, qsnr_db


def read_sigmf_iq_file(
    meta_path: Path | str,
    quantization_scale: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference reader for SigMF datasets (.sigmf-meta + .sigmf-data).

    Args:
        meta_path: Path to .sigmf-meta file.
        quantization_scale: Optional scaling factor for integer dequantization.

    Returns:
        tuple (iq_complex64, metadata_dict)
    """
    path = Path(meta_path)
    if str(path).endswith(".sigmf-data"):
        data_path = path
        meta_path = Path(str(path)[:-11] + ".sigmf-meta")
    else:
        meta_path = path
        data_path = Path(str(path)[:-11] + ".sigmf-data") if str(path).endswith(".sigmf-meta") else path.parent / f"{path.stem}.sigmf-data"

    if not meta_path.is_file():
        raise FileNotFoundError(f"SigMF meta file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if not data_path.is_file():
        raise FileNotFoundError(f"SigMF data file not found: {data_path}")

    sigmf_dtype = meta["global"]["core:datatype"]
    dtype_str, endian = REVERSE_SIGMF_MAP.get(sigmf_dtype, ("float32", "little"))

    iq = read_raw_iq_file(
        file_path=data_path,
        storage_dtype=dtype_str,
        endianness=endian,
        iq_order="IQ",
        quantization_scale=quantization_scale,
    )

    return iq, meta
