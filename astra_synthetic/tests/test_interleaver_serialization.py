"""
Unit tests for InterleaverRecord Serialization and Deserialization in ASTRA Synthetic Engine 4.
Validates individual directory exports, compact batch outputs, numpy array integrity,
JSON metadata structure, and round-trip reloading.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from astra_synthetic.fec.models import FECRecord
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.interleaving.models import InterleaverRecord
from astra_synthetic.interleaving.serializers import (
    load_interleaver_record,
    save_interleaver_batch,
    save_interleaver_record,
)


def make_dummy_fec_record(length: int = 512, seed: int = 42) -> FECRecord:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length, dtype=np.uint8)
    return FECRecord(
        fec_record_id="fec_000001",
        frame_id="frame_000001",
        fec_type="convolutional",
        fec_profile="conv_r1_2_k7",
        input_bits=bits.copy(),
        encoded_bits=bits.copy(),
        input_bit_length=length,
        encoded_bit_length=length,
        nominal_code_rate=1.0,
        effective_code_rate=1.0,
        metadata={"source_payload_id": "pay_000001", "source_frame_id": "frame_000001"},
    )


class TestInterleaverSerialization:
    """Test suite for Interleaver record saving and loading."""

    def test_save_and_load_record_directory(self, tmp_path: Path):
        """Verify record-directory serialization and deserialization."""
        gen = InterleaverGenerator()
        fec_rec = make_dummy_fec_record(length=300)
        int_rec = gen.interleave(fec_rec, profile_name="block_16x32")

        out_dir = tmp_path / "records"
        saved_path = save_interleaver_record(int_rec, output_dir=out_dir, compact=False)

        assert saved_path.is_dir()
        assert (saved_path / "metadata.json").exists()
        assert (saved_path / "input_bits.npy").exists()
        assert (saved_path / "interleaved_bits.npy").exists()
        assert (saved_path / "padding_bits.npy").exists()
        assert (saved_path / "permutation.npy").exists()

        loaded_rec = load_interleaver_record(saved_path)
        assert loaded_rec.interleaver_record_id == int_rec.interleaver_record_id
        assert loaded_rec.interleaver_type == int_rec.interleaver_type
        assert loaded_rec.profile_name == int_rec.profile_name
        assert loaded_rec.input_bit_length == int_rec.input_bit_length
        assert loaded_rec.output_bit_length == int_rec.output_bit_length
        assert loaded_rec.padding_length == int_rec.padding_length
        assert loaded_rec.input_sha256 == int_rec.input_sha256
        assert loaded_rec.output_sha256 == int_rec.output_sha256

        np.testing.assert_array_equal(loaded_rec.input_bits, int_rec.input_bits)
        np.testing.assert_array_equal(loaded_rec.interleaved_bits, int_rec.interleaved_bits)
        np.testing.assert_array_equal(loaded_rec.padding_bits, int_rec.padding_bits)
        if int_rec.permutation is not None:
            np.testing.assert_array_equal(loaded_rec.permutation, int_rec.permutation)

    def test_save_and_load_compact_mode(self, tmp_path: Path):
        """Verify compact serialization mode (bits in interleaved_bits/, meta in metadata/)."""
        gen = InterleaverGenerator()
        fec_rec = make_dummy_fec_record(length=256)
        int_rec = gen.interleave(fec_rec, profile_name="pr_256")

        out_dir = tmp_path / "compact_records"
        saved_path = save_interleaver_record(int_rec, output_dir=out_dir, compact=True)

        assert saved_path.is_file()
        assert saved_path.suffix == ".npy"
        meta_path = out_dir / "metadata" / f"{int_rec.interleaver_record_id}.json"
        assert meta_path.exists()

        loaded_rec = load_interleaver_record(meta_path)
        assert loaded_rec.interleaver_record_id == int_rec.interleaver_record_id
        np.testing.assert_array_equal(loaded_rec.interleaved_bits, int_rec.interleaved_bits)

    def test_save_batch_convenience(self, tmp_path: Path):
        """Verify save_interleaver_batch helper."""
        gen = InterleaverGenerator()
        fec_records = [make_dummy_fec_record(length=200 + i * 20, seed=i) for i in range(4)]
        int_records = gen.interleave_batch(fec_records)

        out_dir = tmp_path / "batch_out"
        saved_paths = save_interleaver_batch(int_records, output_dir=out_dir)
        assert len(saved_paths) == 4
        for p in saved_paths:
            assert p.exists()
