"""
Unit tests for Stereo WAV IQ writer and reader (Engine 7).
"""

from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from astra_synthetic.capture.wav_iq import write_wav_iq_file, read_wav_iq_file


def test_wav_i16_round_trip(tmp_path):
    rng = np.random.default_rng(999)
    x = (rng.standard_normal(400) + 1j * rng.standard_normal(400)).astype(np.complex64)
    sample_rate = 96000.0

    wav_path = tmp_path / "test_stereo.wav"
    out_p, sha, sz, scale, mse, qsnr = write_wav_iq_file(
        iq=x,
        output_path=wav_path,
        sample_rate_hz=sample_rate,
        storage_dtype="int16",
        iq_order="IQ",
    )

    assert out_p.is_file()
    rate, raw_data = wavfile.read(str(out_p))
    assert rate == 96000
    assert raw_data.shape == (400, 2)

    rec_iq, rec_rate = read_wav_iq_file(out_p, iq_order="IQ", quantization_scale=scale)
    assert rec_rate == 96000
    assert len(rec_iq) == 400
    assert np.allclose(x, rec_iq, atol=1e-3)


def test_wav_qi_order(tmp_path):
    x = np.array([1.0 + 5.0j, -2.0 + 8.0j], dtype=np.complex64)
    sample_rate = 48000.0

    wav_path = tmp_path / "test_qi.wav"
    out_p, sha, sz, scale, _, _ = write_wav_iq_file(
        iq=x,
        output_path=wav_path,
        sample_rate_hz=sample_rate,
        storage_dtype="int16",
        iq_order="QI",
    )

    _, raw_data = wavfile.read(str(out_p))
    # In QI order, Ch0 is Q (imag), Ch1 is I (real)
    assert raw_data[0, 0] > 0  # imag is 5.0
    assert raw_data[1, 1] < 0  # real is -2.0

    rec_iq, _ = read_wav_iq_file(out_p, iq_order="QI", quantization_scale=scale)
    assert np.allclose(x, rec_iq, atol=1e-3)
