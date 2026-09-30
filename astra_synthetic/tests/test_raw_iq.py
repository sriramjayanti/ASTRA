"""
Unit tests for Raw IQ file writer, reader, and byte calculations (Engine 7).
"""

from pathlib import Path
import numpy as np
import pytest
from astra_synthetic.capture.raw_iq import write_raw_iq_file, read_raw_iq_file


def test_raw_f32_exact_round_trip(tmp_path):
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)

    file_path = tmp_path / "raw_test.iq"
    written_path, file_sha, file_sz, scale, mse, qsnr = write_raw_iq_file(
        iq=x,
        output_path=file_path,
        storage_dtype="float32",
        endianness="little",
        iq_order="IQ",
    )

    assert written_path.is_file()
    # 512 complex samples * 2 scalars * 4 bytes = 4096 bytes
    assert file_sz == 512 * 2 * 4
    assert written_path.stat().st_size == 4096

    # Float32 is bit-exact round-trip
    recovered = read_raw_iq_file(
        written_path,
        storage_dtype="float32",
        endianness="little",
        iq_order="IQ",
    )
    assert np.array_equal(x, recovered)


def test_raw_i16_round_trip(tmp_path):
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)

    file_path = tmp_path / "int16_test.iq"
    written_path, file_sha, file_sz, scale, mse, qsnr = write_raw_iq_file(
        iq=x,
        output_path=file_path,
        storage_dtype="int16",
        endianness="little",
        iq_order="IQ",
    )

    # 256 complex * 2 scalars * 2 bytes = 1024 bytes
    assert file_sz == 256 * 2 * 2
    assert written_path.stat().st_size == 1024

    recovered = read_raw_iq_file(
        written_path,
        storage_dtype="int16",
        endianness="little",
        iq_order="IQ",
        quantization_scale=scale,
    )
    assert np.allclose(x, recovered, atol=1e-3)
