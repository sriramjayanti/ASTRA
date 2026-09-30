"""
Serialization tests for ASTRA FEC Engine.
Validates disk saving and loading of FECRecord objects and metadata JSON serializability.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.fec.serializers import save_fec_record, load_fec_record, save_fec_batch


class TestFECSerialization:
    """Test disk serialization and deserialization of FEC records."""

    def test_save_and_load_round_trip(self, tmp_path: Path):
        """TEST 14: Serialization/reload preserves entire FEC record."""
        pgen = PayloadGenerator()
        fgen = FrameGenerator()
        fec_gen = FECGenerator()

        payload = pgen.generate(payload_type="random_bits", bit_length=512, seed=42)
        frame = fgen.generate(payload, sequence_number=1)
        record = fec_gen.encode(frame, fec_type="convolutional", profile_name="conv_k7_r12")

        # Save to disk
        fec_dir = save_fec_record(record, output_dir=tmp_path)
        assert fec_dir.is_dir()

        # Verify files exist
        assert (fec_dir / "input_frame_bits.npy").is_file()
        assert (fec_dir / "encoded_bits.npy").is_file()
        assert (fec_dir / "padding_bits.npy").is_file()
        assert (fec_dir / "metadata.json").is_file()

        # TEST 20: Metadata JSON verification
        with open(fec_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["fec_record_id"] == record.fec_record_id
        assert meta["fec_type"] == "convolutional"
        assert meta["fec_profile"] == "conv_k7_r12"

        # Reload record
        reloaded = load_fec_record(fec_dir)
        assert reloaded == record
        assert np.array_equal(reloaded.encoded_bits, record.encoded_bits)
        assert np.array_equal(reloaded.input_bits, record.input_bits)
        assert reloaded.encoded_sha256 == record.encoded_sha256

    def test_save_fec_batch(self, tmp_path: Path):
        pgen = PayloadGenerator()
        fgen = FrameGenerator()
        fec_gen = FECGenerator()

        frames = [
            fgen.generate(pgen.generate(bit_length=256, seed=i), sequence_number=i)
            for i in range(3)
        ]
        records = fec_gen.encode_batch(frames)

        saved_dirs = save_fec_batch(records, output_dir=tmp_path)
        assert len(saved_dirs) == 3
        for d in saved_dirs:
            assert (d / "metadata.json").is_file()
            assert (d / "encoded_bits.npy").is_file()
