"""
Unit tests for Capture metadata pairing and dataset manifest generation (Engine 7).
"""

import csv
import json
from pathlib import Path
import numpy as np
import pytest
from astra_synthetic.capture.models import CaptureRecord
from astra_synthetic.capture.serializers import save_capture_metadata_pair, write_dataset_manifest


@pytest.fixture
def sample_capture_record(tmp_path):
    dummy_file = tmp_path / "cap_test001.iq"
    dummy_file.write_bytes(b"\x00" * 800)  # 100 complex float32 samples = 800 bytes

    return CaptureRecord(
        capture_record_id="cap_test001",
        channel_record_id="chan_test001",
        modulation_record_id="mod_test001",
        interleaver_record_id="int_test001",
        fec_record_id="fec_test001",
        frame_id="frm_test001",
        file_path=str(dummy_file),
        file_format="raw_iq",
        sample_rate_hz=192000.0,
        center_frequency_hz=100000000.0,
        sample_count_complex=100,
        scalar_sample_count=200,
        duration_seconds=100 / 192000.0,
        storage_dtype="float32",
        endianness="little",
        iq_order="IQ",
        channel_count=2,
        quantization_enabled=False,
        file_size_bytes=800,
        file_sha256="abc123sha",
        source_iq_sha256="def456sha",
    )


def test_metadata_pairing_separation(tmp_path, sample_capture_record):
    vis_p, truth_p = save_capture_metadata_pair(sample_capture_record, output_dir=tmp_path)
    assert vis_p.is_file()
    assert truth_p.is_file()

    with open(vis_p, "r", encoding="utf-8") as f:
        vis_dict = json.load(f)

    with open(truth_p, "r", encoding="utf-8") as f:
        truth_dict = json.load(f)

    # Visible dict must NOT expose modulation/SNR/FEC ground truth
    assert "modulation_record_id" not in vis_dict
    assert "channel_record_id" not in vis_dict
    assert vis_dict["sample_rate_hz"] == 192000.0

    # Truth dict contains full lineage
    assert truth_dict["channel_record_id"] == "chan_test001"
    assert truth_dict["modulation_record_id"] == "mod_test001"
    assert truth_dict["sample_rate_hz_truth"] == 192000.0


def test_dataset_manifest_generation(tmp_path, sample_capture_record):
    csv_path = tmp_path / "manifest.csv"
    out_csv = write_dataset_manifest([sample_capture_record], output_csv_path=csv_path)

    assert out_csv.is_file()
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 1
    assert reader[0]["capture_record_id"] == "cap_test001"
    assert reader[0]["file_format"] == "raw_iq"
    assert reader[0]["storage_dtype"] == "float32"
