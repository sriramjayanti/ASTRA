"""
Unit tests for ChannelRecord serialization and deserialization (Engine 6).
"""

from pathlib import Path
import numpy as np
import pytest
from astra_synthetic.channel.models import ChannelRecord
from astra_synthetic.channel.serializers import (
    save_channel_record,
    load_channel_record,
    save_channel_batch,
)


@pytest.fixture
def dummy_channel_record():
    clean = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64)
    impaired = np.array([0.9 + 1.1j, -1.05 + 0.95j, -0.98 - 1.02j, 1.01 - 0.99j], dtype=np.complex64)
    return ChannelRecord(
        channel_record_id="chan_test001",
        modulation_record_id="mod_test001",
        interleaver_record_id="int_test001",
        fec_record_id="fec_test001",
        frame_id="frm_test001",
        clean_iq=clean,
        impaired_iq=impaired,
        sample_rate=192000.0,
        snr_db_target=10.0,
        snr_db_measured=10.02,
        cfo_hz=1500.0,
        phase_offset_rad=0.5,
        timing_offset_samples=0.25,
        timing_offset_symbols=0.03125,
        gain_db=-2.0,
        gain_linear=0.7943,
        frequency_drift_hz_per_sec=10.0,
        fading_type="rician",
        fading_parameters={"k_factor_db": 10.0},
        multipath_enabled=True,
        multipath_taps=np.array([1.0 + 0j, 0.4 + 0.3j], dtype=np.complex64),
        multipath_delays_samples=np.array([0, 3], dtype=np.int64),
        power_stages={"clean_signal": 2.0, "final_power": 1.5},
        impairment_order=["gain", "cfo", "phase_offset", "awgn"],
        parameters={"profile": "test_profile"},
        metadata={"generator": "ASTRA Engine 6"},
    )


def test_channel_serialization_directory_mode(tmp_path, dummy_channel_record):
    saved_dir = save_channel_record(dummy_channel_record, output_dir=tmp_path, compact=False)
    assert saved_dir.is_dir()
    assert (saved_dir / "clean_iq.npy").is_file()
    assert (saved_dir / "impaired_iq.npy").is_file()
    assert (saved_dir / "metadata.json").is_file()

    loaded = load_channel_record(saved_dir)
    assert loaded.channel_record_id == dummy_channel_record.channel_record_id
    assert loaded.modulation_record_id == dummy_channel_record.modulation_record_id
    assert np.allclose(loaded.clean_iq, dummy_channel_record.clean_iq)
    assert np.allclose(loaded.impaired_iq, dummy_channel_record.impaired_iq)
    assert np.isclose(loaded.snr_db_target, dummy_channel_record.snr_db_target)
    assert np.isclose(loaded.cfo_hz, dummy_channel_record.cfo_hz)
    assert loaded.multipath_enabled is True
    assert loaded.clean_iq_sha256 == dummy_channel_record.clean_iq_sha256
    assert loaded.impaired_iq_sha256 == dummy_channel_record.impaired_iq_sha256


def test_channel_serialization_compact_mode(tmp_path, dummy_channel_record):
    saved_file = save_channel_record(dummy_channel_record, output_dir=tmp_path, compact=True)
    assert saved_file.is_file()

    meta_file = tmp_path / "metadata" / f"{dummy_channel_record.channel_record_id}.json"
    assert meta_file.is_file()

    loaded = load_channel_record(meta_file)
    assert loaded.channel_record_id == dummy_channel_record.channel_record_id
    assert np.allclose(loaded.impaired_iq, dummy_channel_record.impaired_iq)
    assert loaded.impaired_iq_sha256 == dummy_channel_record.impaired_iq_sha256


def test_channel_batch_save(tmp_path, dummy_channel_record):
    records = [dummy_channel_record]
    saved_paths = save_channel_batch(records, output_dir=tmp_path, compact=False)
    assert len(saved_paths) == 1
    assert saved_paths[0].is_dir()
