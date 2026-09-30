"""
Integration and functional tests for ModulationGenerator in ASTRA Synthetic Engine 5.
Validates InterleaverRecord consumption, immutability, generator API, profile sampling,
all modulation families, metadata lineage preservation, and reference demodulation.
"""

import numpy as np
import pytest

from astra_synthetic.interleaving.models import InterleaverRecord
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.modulation.models import ModulationRecord
from astra_synthetic.modulation.validators import reference_demodulate, validate_modulation_record


def make_dummy_interleaver_record(length: int = 512, seed: int = 42) -> InterleaverRecord:
    """Helper to create a realistic dummy InterleaverRecord."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length, dtype=np.uint8)
    return InterleaverRecord(
        interleaver_record_id="int_000001",
        fec_record_id="fec_000001",
        frame_id="frame_000001",
        interleaver_type="block",
        profile_name="block_16x32",
        input_bits=bits.copy(),
        interleaved_bits=bits.copy(),
        input_bit_length=length,
        output_bit_length=length,
        parameters={"rows": 16, "columns": 32},
        metadata={
            "source_payload_id": "pay_000001",
            "source_frame_id": "frame_000001",
            "source_fec_record_id": "fec_000001",
        },
    )


class TestModulationGenerator:
    """Integration test suite for ModulationGenerator."""

    def test_interleaver_record_immutability(self):
        """Verify generator never mutates the input InterleaverRecord bit array."""
        gen = ModulationGenerator()
        int_rec = make_dummy_interleaver_record(length=300)
        original_copy = int_rec.interleaved_bits.copy()

        for mtype in ["2fsk", "4fsk", "bpsk", "qpsk", "8psk", "16qam", "64qam"]:
            mod_rec = gen.modulate(int_rec, modulation_type=mtype)
            np.testing.assert_array_equal(int_rec.interleaved_bits, original_copy)
            assert int_rec.interleaver_record_id == "int_000001"

    def test_all_builtin_profiles_roundtrip(self):
        """Verify every built-in modulation profile generates valid records and passes reference_demodulate."""
        gen = ModulationGenerator()
        int_rec = make_dummy_interleaver_record(length=600, seed=777)

        for profile_name in gen.profiles.keys():
            mod_rec = gen.modulate(int_rec, profile_name=profile_name)
            validate_modulation_record(mod_rec, original_interleaver=int_rec)

            # Output constraints
            assert mod_rec.clean_iq.dtype == np.complex64
            assert np.all(np.isfinite(mod_rec.clean_iq))
            assert abs(mod_rec.average_iq_power - 1.0) < 0.05

            # Reference demodulation exact round-trip
            recovered = reference_demodulate(mod_rec)
            np.testing.assert_array_equal(recovered, int_rec.interleaved_bits)

    def test_lineage_metadata_preservation(self):
        """Verify source payload, frame, FEC, and interleaver IDs are preserved for ML train/test splitting."""
        gen = ModulationGenerator()
        int_rec = make_dummy_interleaver_record(length=128)
        mod_rec = gen.modulate(int_rec, profile_name="qpsk_rrc")

        assert mod_rec.interleaver_record_id == "int_000001"
        assert mod_rec.fec_record_id == "fec_000001"
        assert mod_rec.frame_id == "frame_000001"
        assert mod_rec.metadata["source_payload_id"] == "pay_000001"
        assert mod_rec.metadata["source_frame_id"] == "frame_000001"
        assert mod_rec.metadata["source_fec_record_id"] == "fec_000001"
        assert mod_rec.metadata["source_interleaver_record_id"] == "int_000001"

    def test_batch_and_iter_modulated(self):
        """Verify batch generation and streaming iterator."""
        gen = ModulationGenerator()
        int_records = [make_dummy_interleaver_record(length=200 + i * 40, seed=i) for i in range(5)]

        # Batch
        batch_out = gen.modulate_batch(int_records)
        assert len(batch_out) == 5

        # Streaming
        stream_out = list(gen.iter_modulated(int_records))
        assert len(stream_out) == 5

        for rec in stream_out:
            validate_modulation_record(rec)

    def test_deterministic_seeded_generation(self):
        """Verify master seed produces identical outputs across generator instances."""
        int_rec = make_dummy_interleaver_record(length=256, seed=88)
        gen1 = ModulationGenerator(seed=42)
        gen2 = ModulationGenerator(seed=42)

        rec1 = gen1.modulate(int_rec)
        rec2 = gen2.modulate(int_rec)

        assert rec1.modulation_type == rec2.modulation_type
        assert rec1.pulse_shape == rec2.pulse_shape
        np.testing.assert_array_equal(rec1.clean_iq, rec2.clean_iq)
        assert rec1.iq_sha256 == rec2.iq_sha256

    def test_invalid_profile_error(self):
        """Verify error is raised when an unknown profile name is requested."""
        gen = ModulationGenerator()
        int_rec = make_dummy_interleaver_record(length=128)
        with pytest.raises(ValueError, match="Unknown modulation profile"):
            gen.modulate(int_rec, profile_name="non_existent_profile_abc")
