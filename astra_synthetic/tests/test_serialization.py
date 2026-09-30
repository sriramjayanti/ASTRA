"""
Serialization and file persistence tests for ASTRA Payload Engine.
Validates round-trip saving and loading across multi-format outputs (.bin, .npy, .json, .txt).
"""

import json
from pathlib import Path
import numpy as np
import pytest

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.payload.serializers import save_payload, load_payload, save_batch


class TestSerialization:
    """Test disk serialization and deserialization."""

    def test_save_and_load_round_trip(self, tmp_path: Path):
        """TEST 13: serialization and reload preserves bits exactly"""
        gen = PayloadGenerator()
        record = gen.generate(
            payload_type="random_bits",
            bit_length=1024,
            seed=42,
            payload_id="payload_000042",
        )

        # Save record
        payload_dir = save_payload(record, output_dir=tmp_path)
        assert payload_dir.is_dir()

        # Check expected files exist
        assert (payload_dir / "payload.bin").is_file()
        assert (payload_dir / "payload_bits.npy").is_file()
        assert (payload_dir / "payload.txt").is_file()
        assert (payload_dir / "metadata.json").is_file()

        # Check metadata json content
        with open(payload_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["payload_id"] == "payload_000042"
        assert meta["bit_length"] == 1024
        assert meta["sha256"] == record.sha256

        # Reload record
        loaded_record = load_payload(payload_dir)
        assert loaded_record == record
        assert np.array_equal(loaded_record.payload_bits, record.payload_bits)
        assert loaded_record.payload_bytes == record.payload_bytes
        assert loaded_record.sha256 == record.sha256

    def test_save_batch(self, tmp_path: Path):
        gen = PayloadGenerator()
        records = gen.generate_batch(count=5)
        saved_paths = save_batch(records, output_dir=tmp_path)
        assert len(saved_paths) == 5
        for p in saved_paths:
            assert p.is_dir()
            assert (p / "metadata.json").is_file()

    def test_text_file_omission_for_huge_payloads(self, tmp_path: Path):
        gen = PayloadGenerator()
        huge_rec = gen.generate(
            payload_type="random_bits",
            bit_length=16384,
            seed=1,
            payload_id="payload_huge",
        )
        saved_dir = save_payload(huge_rec, output_dir=tmp_path, max_txt_bits=8192)
        # Should not write payload.txt because bit_length > max_txt_bits
        assert not (saved_dir / "payload.txt").exists()
        assert (saved_dir / "payload_bits.npy").exists()
        assert (saved_dir / "payload.bin").exists()
