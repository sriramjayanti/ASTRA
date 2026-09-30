"""
Integration and functional tests for InterleaverGenerator in ASTRA Synthetic Engine 4.
Validates FECRecord consumption, immutability, generator API, profile sampling,
all interleaver families, metadata lineage preservation, and reference deinterleaving.
"""

import numpy as np
import pytest

from astra_synthetic.fec.models import FECRecord
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.interleaving.models import InterleaverRecord
from astra_synthetic.interleaving.validators import reference_deinterleave, validate_interleaver_record


def make_dummy_fec_record(length: int = 512, fec_type: str = "convolutional", seed: int = 42) -> FECRecord:
    """Helper to create a realistic dummy FECRecord."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length, dtype=np.uint8)
    return FECRecord(
        fec_record_id="fec_000001",
        frame_id="frame_000001",
        fec_type=fec_type,
        fec_profile="conv_r1_2_k7",
        input_bits=bits.copy(),
        encoded_bits=bits.copy(),
        input_bit_length=length,
        encoded_bit_length=length,
        nominal_code_rate=1.0,
        effective_code_rate=1.0,
        metadata={"source_payload_id": "pay_000001", "source_frame_id": "frame_000001"},
    )


class TestInterleaverGenerator:
    """Integration test suite for InterleaverGenerator."""

    def test_none_interleaver_identity(self):
        """Verify 'none' interleaver produces an exact identity copy with 0 padding."""
        gen = InterleaverGenerator()
        fec_rec = make_dummy_fec_record(length=256)
        int_rec = gen.interleave(fec_rec, interleaver_type="none")

        assert int_rec.interleaver_type == "none"
        assert int_rec.input_bit_length == 256
        assert int_rec.output_bit_length == 256
        assert int_rec.padding_length == 0
        np.testing.assert_array_equal(int_rec.interleaved_bits, fec_rec.encoded_bits)
        assert int_rec.input_sha256 == int_rec.output_sha256

        # Reference deinterleaving
        recovered = reference_deinterleave(int_rec)
        np.testing.assert_array_equal(recovered, fec_rec.encoded_bits)

    def test_fecrecord_immutability(self):
        """Verify generator never alters or mutates the input FECRecord bits."""
        gen = InterleaverGenerator()
        fec_rec = make_dummy_fec_record(length=300)
        original_copy = fec_rec.encoded_bits.copy()

        # Run multiple interleavers
        for itype in ["none", "block", "convolutional", "diagonal", "pseudo_random"]:
            int_rec = gen.interleave(fec_rec, interleaver_type=itype)
            # Ensure FECRecord remains untouched
            np.testing.assert_array_equal(fec_rec.encoded_bits, original_copy)
            assert fec_rec.fec_record_id == "fec_000001"

    def test_all_builtin_profiles_roundtrip(self):
        """Verify every built-in profile produces valid records that pass reference_deinterleave."""
        gen = InterleaverGenerator()
        fec_rec = make_dummy_fec_record(length=600, seed=1234)

        for profile_name in gen.profiles.keys():
            int_rec = gen.interleave(fec_rec, profile_name=profile_name)
            validate_interleaver_record(int_rec)

            # Output array constraints
            assert int_rec.interleaved_bits.dtype == np.uint8
            assert np.all(np.isin(int_rec.interleaved_bits, [0, 1]))

            # Exact recovery
            recovered = reference_deinterleave(int_rec)
            np.testing.assert_array_equal(recovered, fec_rec.encoded_bits)

    def test_lineage_metadata_preservation(self):
        """Verify source_payload_id, frame_id, and fec_record_id are preserved for ML splitting."""
        gen = InterleaverGenerator()
        fec_rec = make_dummy_fec_record(length=128)
        int_rec = gen.interleave(fec_rec, profile_name="block_8x8")

        assert int_rec.fec_record_id == "fec_000001"
        assert int_rec.frame_id == "frame_000001"
        assert int_rec.metadata["source_payload_id"] == "pay_000001"
        assert int_rec.metadata["source_frame_id"] == "frame_000001"
        assert int_rec.metadata["source_fec_record_id"] == "fec_000001"

    def test_batch_and_iter_interleaved(self):
        """Verify batch generation and generator iterator."""
        gen = InterleaverGenerator()
        fec_records = [make_dummy_fec_record(length=200 + i * 50, seed=i) for i in range(5)]

        # Batch
        batch_out = gen.interleave_batch(fec_records)
        assert len(batch_out) == 5

        # Streaming
        stream_out = list(gen.iter_interleaved(fec_records))
        assert len(stream_out) == 5

        for rec in stream_out:
            validate_interleaver_record(rec)

    def test_deterministic_seeded_generation(self):
        """Verify master seed produces identical outputs across instances."""
        fec_rec = make_dummy_fec_record(length=350, seed=77)
        gen1 = InterleaverGenerator(seed=42)
        gen2 = InterleaverGenerator(seed=42)

        rec1 = gen1.interleave(fec_rec)
        rec2 = gen2.interleave(fec_rec)

        assert rec1.interleaver_type == rec2.interleaver_type
        assert rec1.profile_name == rec2.profile_name
        np.testing.assert_array_equal(rec1.interleaved_bits, rec2.interleaved_bits)

    def test_invalid_profile_error(self):
        """Verify error is raised when an unknown profile name is requested."""
        gen = InterleaverGenerator()
        fec_rec = make_dummy_fec_record(length=128)
        with pytest.raises(ValueError, match="Unknown interleaver profile"):
            gen.interleave(fec_rec, profile_name="non_existent_profile_xyz")
