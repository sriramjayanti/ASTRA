"""
Serialization tests for ASTRA Framing Engine.
Validates round-trip saving and loading of FrameRecord and FrameStreamRecord objects.
"""

from pathlib import Path
import numpy as np
import pytest

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.framing.serializers import (
    save_frame,
    load_frame,
    save_frame_batch,
    save_frame_stream,
    load_frame_stream,
)


class TestFrameSerialization:
    """Test disk serialization of individual frames and streams."""

    def test_frame_save_and_load_round_trip(self, tmp_path: Path):
        """TEST 16: serialization/reload preserves entire frame"""
        pgen = PayloadGenerator()
        fgen = FrameGenerator()

        payload = pgen.generate(payload_type="random_bits", bit_length=1024, seed=42)
        frame = fgen.generate(payload, sequence_number=42)

        # Save to disk
        frame_dir = save_frame(frame, output_dir=tmp_path)
        assert frame_dir.is_dir()

        # Check expected files
        assert (frame_dir / "frame.bin").is_file()
        assert (frame_dir / "frame_bits.npy").is_file()
        assert (frame_dir / "frame_bits.txt").is_file()
        assert (frame_dir / "sync_bits.npy").is_file()
        assert (frame_dir / "header_bits.npy").is_file()
        assert (frame_dir / "payload_bits.npy").is_file()
        assert (frame_dir / "crc_bits.npy").is_file()
        assert (frame_dir / "metadata.json").is_file()

        # Reload frame
        reloaded = load_frame(frame_dir)
        assert reloaded == frame
        assert np.array_equal(reloaded.frame_bits, frame.frame_bits)
        assert reloaded.sha256 == frame.sha256
        assert reloaded.frame_sequence_number == 42

    def test_frame_batch_saving(self, tmp_path: Path):
        pgen = PayloadGenerator()
        fgen = FrameGenerator()

        payloads = pgen.generate_batch(count=3)
        frames = fgen.generate_batch(payloads)

        saved_dirs = save_frame_batch(frames, output_dir=tmp_path)
        assert len(saved_dirs) == 3
        for d in saved_dirs:
            assert (d / "metadata.json").is_file()
            assert (d / "frame_bits.npy").is_file()

    def test_frame_stream_save_and_load(self, tmp_path: Path):
        pgen = PayloadGenerator()
        fgen = FrameGenerator()

        payloads = [pgen.generate(payload_type="random_bits", bit_length=256, seed=i) for i in range(4)]
        stream = fgen.generate_stream(payloads, inter_frame_gap={"enabled": True, "length_bits": 16, "mode": "zeros"})

        npy_path, meta_path = save_frame_stream(stream, output_dir=tmp_path, base_name="test_stream")
        assert npy_path.is_file()
        assert meta_path.is_file()

        loaded_stream = load_frame_stream(npy_path, meta_path)
        assert np.array_equal(loaded_stream.stream_bits, stream.stream_bits)
        assert loaded_stream.total_bit_length == stream.total_bit_length
        assert loaded_stream.frame_ids == stream.frame_ids
        assert loaded_stream.frame_start_positions == stream.frame_start_positions
        assert loaded_stream.sha256 == stream.sha256
