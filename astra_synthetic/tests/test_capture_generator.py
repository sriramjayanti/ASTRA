"""
Integration tests for CaptureGenerator across the complete 7-engine pipeline.
"""

from pathlib import Path
import numpy as np
import pytest

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator
from astra_synthetic.capture.generator import CaptureGenerator
from astra_synthetic.capture.raw_iq import read_raw_iq_file
from astra_synthetic.capture.wav_iq import read_wav_iq_file


@pytest.fixture
def sample_channel_record():
    p_gen = PayloadGenerator()
    f_gen = FrameGenerator()
    fec_gen = FECGenerator()
    int_gen = InterleaverGenerator()
    mod_gen = ModulationGenerator()
    chan_gen = ChannelGenerator()

    p_rec = p_gen.generate(payload_type="counter", bit_length=256, seed=42)
    f_rec = f_gen.generate(p_rec)
    fec_rec = fec_gen.encode(f_rec, scheme="none")
    int_rec = int_gen.interleave(fec_rec, scheme="none")
    mod_rec = mod_gen.modulate(int_rec, modulation_type="qpsk", samples_per_symbol=8)
    chan_rec = chan_gen.apply(mod_rec, profile_name="combined_medium", seed=1001)
    return chan_rec


def test_capture_generator_raw_f32_pipeline(tmp_path, sample_channel_record):
    cap_gen = CaptureGenerator()
    orig_iq_copy = sample_channel_record.impaired_iq.copy()

    rec = cap_gen.write(
        sample_channel_record,
        profile_name="raw_f32_le_iq",
        output_dir=tmp_path,
    )

    # Immutability check: original ChannelRecord was not mutated
    assert np.array_equal(sample_channel_record.impaired_iq, orig_iq_copy)

    assert rec.file_format == "raw_iq"
    assert rec.storage_dtype == "float32"
    assert rec.channel_record_id == sample_channel_record.channel_record_id
    assert rec.modulation_record_id == sample_channel_record.modulation_record_id
    assert rec.sample_count_complex == len(sample_channel_record.impaired_iq)
    assert Path(rec.file_path).is_file()

    # Verify visible and truth metadata files exist
    assert Path(rec.visible_metadata_path).is_file()
    assert Path(rec.truth_metadata_path).is_file()


def test_capture_generator_wav_pipeline(tmp_path, sample_channel_record):
    cap_gen = CaptureGenerator()
    rec = cap_gen.write(
        sample_channel_record,
        profile_name="wav_i16_iq",
        output_dir=tmp_path,
    )

    assert rec.file_format == "wav"
    assert rec.storage_dtype == "int16"
    assert rec.quantization_enabled is True
    assert Path(rec.file_path).is_file()


def test_capture_generator_missing_samplerate_masking(tmp_path, sample_channel_record):
    cap_gen = CaptureGenerator()
    rec = cap_gen.write(
        sample_channel_record,
        profile_name="missing_samplerate_test",
        output_dir=tmp_path,
    )

    assert rec.sample_rate_visible is False
    visible_dict = rec.to_visible_dict()
    assert visible_dict["sample_rate_hz"] is None

    truth_dict = rec.to_truth_dict()
    assert truth_dict["sample_rate_hz_truth"] == sample_channel_record.sample_rate


def test_capture_generator_batch_and_iter(tmp_path, sample_channel_record):
    cap_gen = CaptureGenerator()
    records = [sample_channel_record, sample_channel_record]

    batch = cap_gen.write_batch(records, profile_name="raw_f32_le_iq", output_dir=tmp_path)
    assert len(batch) == 2
    assert batch[0].capture_record_id != batch[1].capture_record_id
    assert Path(batch[0].file_path).is_file()
    assert Path(batch[1].file_path).is_file()
