"""
Unit tests for Endianness handling and binary byte ordering (Engine 7).
"""

from pathlib import Path
import numpy as np
import pytest
from astra_synthetic.capture.raw_iq import write_raw_iq_file, read_raw_iq_file


def test_endianness_byte_divergence(tmp_path):
    # Known signal with distinct values
    x = np.array([1.0 + 2.0j], dtype=np.complex64)

    p_le = tmp_path / "test_le.iq"
    p_be = tmp_path / "test_be.iq"

    # Write little-endian float32
    write_raw_iq_file(x, p_le, storage_dtype="float32", endianness="little", iq_order="IQ")
    # Write big-endian float32
    write_raw_iq_file(x, p_be, storage_dtype="float32", endianness="big", iq_order="IQ")

    with open(p_le, "rb") as f:
        bytes_le = f.read()
    with open(p_be, "rb") as f:
        bytes_be = f.read()

    # Raw bytes must differ for multi-byte types
    assert bytes_le != bytes_be
    # First float32 (1.0): LE is 00 00 80 3f, BE is 3f 80 00 00
    assert bytes_le[0:4] == bytes_be[0:4][::-1]

    # Both readers must accurately recover the identical numerical values
    rec_le = read_raw_iq_file(p_le, storage_dtype="float32", endianness="little", iq_order="IQ")
    rec_be = read_raw_iq_file(p_be, storage_dtype="float32", endianness="big", iq_order="IQ")

    assert np.allclose(x, rec_le)
    assert np.allclose(x, rec_be)


def test_endianness_int16(tmp_path):
    x = np.array([1000.0 + -2000.0j], dtype=np.complex64)

    p_le = tmp_path / "int16_le.iq"
    p_be = tmp_path / "int16_be.iq"

    _, _, _, scale_le, _, _ = write_raw_iq_file(
        x, p_le, storage_dtype="int16", endianness="little", iq_order="IQ"
    )
    _, _, _, scale_be, _, _ = write_raw_iq_file(
        x, p_be, storage_dtype="int16", endianness="big", iq_order="IQ"
    )

    with open(p_le, "rb") as f:
        bytes_le = f.read()
    with open(p_be, "rb") as f:
        bytes_be = f.read()

    assert bytes_le != bytes_be

    rec_le = read_raw_iq_file(p_le, storage_dtype="int16", endianness="little", iq_order="IQ", quantization_scale=scale_le)
    rec_be = read_raw_iq_file(p_be, storage_dtype="int16", endianness="big", iq_order="IQ", quantization_scale=scale_be)

    assert np.allclose(x, rec_le, rtol=1e-3)
    assert np.allclose(x, rec_be, rtol=1e-3)
