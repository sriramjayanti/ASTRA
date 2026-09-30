"""
Unit tests for ModulationRecord Serialization and Deserialization in ASTRA Synthetic Engine 5.
"""

from pathlib import Path
import numpy as np
import pytest

from astra_synthetic.interleaving.models import InterleaverRecord
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.modulation.models import ModulationRecord
from astra_synthetic.modulation.serializers import (
    load_modulation_record,
    save_modulation_batch,
    save_modulation_record,
)


def make_dummy_interleaver_record(length: int = 512, seed: int = 42) -> InterleaverRecord:
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


class TestModulationSerialization:
    """Test suite for Modulation record saving and loading."""

    def test_save_and_load_record_directory(self, tmp_path: Path):
        """Verify record-directory serialization and deserialization."""
        gen = ModulationGenerator()
        int_rec = make_dummy_interleaver_record(length=300)
        mod_rec = gen.modulate(int_rec, profile_name="16qam_rrc")

        out_dir = tmp_path / "records"
        saved_path = save_modulation_record(mod_rec, output_dir=out_dir, compact=False)

        assert saved_path.is_dir()
        assert (saved_path / "metadata.json").exists()
        assert (saved_path / "input_bits.npy").exists()
        assert (saved_path / "symbol_indices.npy").exists()
        assert (saved_path / "ideal_symbols.npy").exists()
        assert (saved_path / "clean_iq.npy").exists()

        loaded_rec = load_modulation_record(saved_path)
        assert loaded_rec.modulation_record_id == mod_rec.modulation_record_id
        assert loaded_rec.modulation_type == mod_rec.modulation_type
        assert loaded_rec.modulation_family == mod_rec.modulation_family
        assert loaded_rec.bits_per_symbol == mod_rec.bits_per_symbol
        assert loaded_rec.symbol_count == mod_rec.symbol_count
        assert loaded_rec.clean_iq_sample_count == mod_rec.clean_iq_sample_count
        assert loaded_rec.input_sha256 == mod_rec.input_sha256
        assert loaded_rec.iq_sha256 == mod_rec.iq_sha256

        np.testing.assert_array_equal(loaded_rec.input_bits, mod_rec.input_bits)
        np.testing.assert_array_equal(loaded_rec.symbol_indices, mod_rec.symbol_indices)
        np.testing.assert_allclose(loaded_rec.ideal_symbols, mod_rec.ideal_symbols, atol=1e-6)
        np.testing.assert_allclose(loaded_rec.clean_iq, mod_rec.clean_iq, atol=1e-6)

    def test_save_and_load_compact_mode(self, tmp_path: Path):
        """Verify compact serialization mode (clean_iq in clean_iq/, meta in metadata/)."""
        gen = ModulationGenerator()
        int_rec = make_dummy_interleaver_record(length=256)
        mod_rec = gen.modulate(int_rec, profile_name="qpsk_rrc")

        out_dir = tmp_path / "compact_records"
        saved_path = save_modulation_record(mod_rec, output_dir=out_dir, compact=True)

        assert saved_path.is_file()
        assert saved_path.suffix == ".npy"
        meta_path = out_dir / "metadata" / f"{mod_rec.modulation_record_id}.json"
        assert meta_path.exists()

        loaded_rec = load_modulation_record(meta_path)
        assert loaded_rec.modulation_record_id == mod_rec.modulation_record_id
        np.testing.assert_allclose(loaded_rec.clean_iq, mod_rec.clean_iq, atol=1e-6)

    def test_save_batch_convenience(self, tmp_path: Path):
        """Verify save_modulation_batch helper."""
        gen = ModulationGenerator()
        int_records = [make_dummy_interleaver_record(length=200 + i * 20, seed=i) for i in range(4)]
        mod_records = gen.modulate_batch(int_records)

        out_dir = tmp_path / "batch_out"
        saved_paths = save_modulation_batch(mod_records, output_dir=out_dir)
        assert len(saved_paths) == 4
        for p in saved_paths:
            assert p.exists()
