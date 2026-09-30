"""
End-to-end integration tests for ASTRA FECGenerator.
Validates No-FEC identity, profile sampling, FrameRecord immutability, reference decoding, and error handling.
"""

import copy
import hashlib
import numpy as np
import pytest

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.fec.validators import reference_decode, validate_fec_record


class TestFECGenerator:
    """Test full FEC generation pipeline and FrameRecord ground-truth integration."""

    @pytest.fixture
    def test_frame(self):
        pgen = PayloadGenerator()
        fgen = FrameGenerator()
        payload = pgen.generate(payload_type="random_bits", bit_length=512, seed=42, payload_id="payload_000001")
        return fgen.generate(payload, sequence_number=1, frame_id="frame_000001")

    def test_no_fec_identity(self, test_frame):
        """TEST 1: No-FEC identity mode."""
        fec_gen = FECGenerator()
        rec = fec_gen.encode(test_frame, fec_type="none")

        assert rec.fec_type == "none"
        assert rec.effective_code_rate == 1.0
        assert rec.nominal_code_rate == 1.0
        assert rec.padding_length == 0
        assert np.array_equal(rec.encoded_bits, test_frame.frame_bits)
        assert np.array_equal(rec.input_bits, test_frame.frame_bits)

    def test_framerecord_immutability(self, test_frame):
        """TEST 17: No mutation of FrameRecord during encoding."""
        fec_gen = FECGenerator()
        orig_bits_copy = test_frame.frame_bits.copy()
        orig_sha = test_frame.sha256

        _ = fec_gen.encode(test_frame, fec_type="convolutional")
        _ = fec_gen.encode(test_frame, fec_type="reed_solomon")
        _ = fec_gen.encode(test_frame, fec_type="ldpc")

        assert np.array_equal(test_frame.frame_bits, orig_bits_copy)
        assert test_frame.sha256 == orig_sha

    def test_all_fec_families_reference_decode(self, test_frame):
        """Test that all 5 FEC families decode back bit-exact to the original FrameRecord."""
        fec_gen = FECGenerator()
        families = ["none", "convolutional", "reed_solomon", "concatenated", "ldpc"]

        for fam in families:
            rec = fec_gen.encode(test_frame, fec_type=fam)
            # Check bit properties (TEST 18 & 19)
            assert rec.encoded_bits.dtype == np.uint8
            assert rec.input_bits.dtype == np.uint8
            assert set(np.unique(rec.encoded_bits)).issubset({0, 1})

            # Check hash integrity (TEST 13)
            expected_in_sha = hashlib.sha256(np.packbits(rec.input_bits).tobytes()).hexdigest()
            assert rec.input_sha256 == expected_in_sha

            # Reference decode round-trip
            decoded = reference_decode(rec)
            assert np.array_equal(decoded, test_frame.frame_bits)

    def test_deterministic_generation_with_seed(self, test_frame):
        """TEST 15: Deterministic profile selection and reproducible encoding."""
        fec_gen1 = FECGenerator({"fec_generator": {"master_seed": 777}})
        fec_gen2 = FECGenerator({"fec_generator": {"master_seed": 777}})

        rec1 = fec_gen1.encode(test_frame, fec_record_id="fec_000001")
        rec2 = fec_gen2.encode(test_frame, fec_record_id="fec_000001")

        assert rec1 == rec2
        assert rec1.fec_type == rec2.fec_type
        assert rec1.fec_profile == rec2.fec_profile
        assert np.array_equal(rec1.encoded_bits, rec2.encoded_bits)

    def test_invalid_profile_raises_error(self, test_frame):
        """TEST 16: Invalid profile raises error."""
        fec_gen = FECGenerator()
        with pytest.raises(ValueError, match="Unknown FEC profile"):
            fec_gen.encode(test_frame, profile_name="quantum_hyper_code_9000")

    def test_streaming_and_batch_generation(self, test_frame):
        """Test iter_encoded and encode_batch streaming mode."""
        pgen = PayloadGenerator()
        fgen = FrameGenerator()
        fec_gen = FECGenerator()

        frames = [
            fgen.generate(pgen.generate(bit_length=256, seed=i, payload_id=f"p_{i}"), sequence_number=i)
            for i in range(5)
        ]

        batch = fec_gen.encode_batch(frames, balanced=True)
        assert len(batch) == 5
        types_used = [b.fec_type for b in batch]
        # In balanced mode across 5 items, all 5 enabled types are represented
        assert set(types_used) == {"none", "convolutional", "reed_solomon", "concatenated", "ldpc"}
