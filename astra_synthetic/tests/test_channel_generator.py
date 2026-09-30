"""
Integration tests for ChannelGenerator and pipeline execution (Engine 6).
"""

import numpy as np
import pytest
from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator


@pytest.fixture
def sample_mod_record():
    p_gen = PayloadGenerator()
    f_gen = FrameGenerator()
    fec_gen = FECGenerator()
    int_gen = InterleaverGenerator()
    mod_gen = ModulationGenerator()

    p_rec = p_gen.generate(payload_type="counter", bit_length=128, seed=42)
    f_rec = f_gen.generate(p_rec)
    fec_rec = fec_gen.encode(f_rec, scheme="none")
    int_rec = int_gen.interleave(fec_rec, scheme="none")
    mod_rec = mod_gen.modulate(int_rec, modulation_type="qpsk", samples_per_symbol=8)
    return mod_rec


def test_channel_generator_clean_profile(sample_mod_record):
    chan_gen = ChannelGenerator()
    clean_copy = sample_mod_record.clean_iq.copy()

    chan_rec = chan_gen.apply(sample_mod_record, profile_name="clean")

    # Clean IQ should not be mutated
    assert np.array_equal(sample_mod_record.clean_iq, clean_copy)
    assert np.array_equal(chan_rec.clean_iq, sample_mod_record.clean_iq)
    assert np.array_equal(chan_rec.impaired_iq, sample_mod_record.clean_iq)
    assert chan_rec.fading_type == "none"
    assert not chan_rec.multipath_enabled
    assert chan_rec.cfo_hz == 0.0


def test_channel_generator_combined_medium(sample_mod_record):
    chan_gen = ChannelGenerator()
    chan_rec = chan_gen.apply(sample_mod_record, profile_name="combined_medium", seed=1001)

    assert chan_rec.channel_record_id.startswith("chan_")
    assert chan_rec.modulation_record_id == sample_mod_record.modulation_record_id
    assert chan_rec.interleaver_record_id == sample_mod_record.interleaver_record_id
    assert chan_rec.fec_record_id == sample_mod_record.fec_record_id
    assert chan_rec.frame_id == sample_mod_record.frame_id

    assert chan_rec.impaired_iq.dtype == np.complex64
    assert np.all(np.isfinite(chan_rec.impaired_iq))
    assert chan_rec.snr_db_measured is not None
    assert chan_rec.multipath_enabled is True
    assert len(chan_rec.impairment_order) > 0


def test_channel_generator_batch_and_iter(sample_mod_record):
    chan_gen = ChannelGenerator()
    records = [sample_mod_record, sample_mod_record]

    batch = chan_gen.apply_batch(records, profile_name="satellite_like_v1", base_seed=555)
    assert len(batch) == 2
    assert batch[0].channel_record_id != batch[1].channel_record_id
    # Different seeds applied per item
    assert not np.array_equal(batch[0].impaired_iq, batch[1].impaired_iq)
