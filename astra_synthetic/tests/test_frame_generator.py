"""
Core functional tests for ASTRA FrameGenerator.
Validates complete frame assembly, ground-truth region mapping, deterministic repeatability,
variable length handling, and truth-parsing recovery.
"""

import hashlib
import numpy as np
import pytest

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.framing.validators import parse_known_frame, validate_frame_record


class TestFrameGenerator:
    """Test full framing pipeline and ground-truth boundary preservation."""

    @pytest.fixture
    def payload_generator(self):
        return PayloadGenerator()

    @pytest.fixture
    def frame_generator(self):
        return FrameGenerator()

    def test_frame_layout_and_slices(self, payload_generator, frame_generator):
        """TEST 1, TEST 2, TEST 3, TEST 10: frame layout, sync position, payload preservation, region boundaries"""
        payload = payload_generator.generate(payload_type="random_bits", bit_length=1024, seed=42)
        frame = frame_generator.generate(payload)

        # Total length check
        expected_len = frame.sync_length + frame.header_length + frame.payload_length + frame.crc_length
        assert frame.frame_bit_length == expected_len
        assert len(frame.frame_bits) == expected_len

        # TEST 3: Sync at start = 0
        assert frame.sync_start == 0
        assert frame.sync_length == 32
        assert np.array_equal(frame.frame_bits[0:32], frame.sync_bits)

        # Header right after sync
        assert frame.header_start == 32
        assert frame.header_length == 56
        assert np.array_equal(frame.frame_bits[32 : 32 + 56], frame.header_bits)

        # TEST 2: Payload preserved exactly
        assert frame.payload_start == 88
        assert frame.payload_length == 1024
        assert np.array_equal(frame.frame_bits[88 : 88 + 1024], payload.payload_bits)

        # CRC at the end
        assert frame.crc_start == 1112
        assert frame.crc_length == 16
        assert np.array_equal(frame.frame_bits[1112 : 1112 + 16], frame.crc_bits)

    def test_deterministic_generation(self, payload_generator):
        """TEST 11: same configuration gives deterministic results"""
        fg1 = FrameGenerator({"frame_generator": {"master_seed": 999}})
        fg2 = FrameGenerator({"frame_generator": {"master_seed": 999}})

        p1 = payload_generator.generate(payload_type="random_bits", bit_length=512, seed=123, payload_id="payload_const")
        p2 = payload_generator.generate(payload_type="random_bits", bit_length=512, seed=123, payload_id="payload_const")

        f1 = fg1.generate(p1, sequence_number=10)
        f2 = fg2.generate(p2, sequence_number=10)

        assert f1 == f2
        assert np.array_equal(f1.frame_bits, f2.frame_bits)
        assert f1.sha256 == f2.sha256

    def test_variable_payload_lengths(self, payload_generator, frame_generator):
        """TEST 14: variable payload lengths supported"""
        for bit_len in [64, 256, 1024, 4096]:
            payload = payload_generator.generate(payload_type="random_bits", bit_length=bit_len, seed=bit_len)
            frame = frame_generator.generate(payload)

            assert frame.payload_length == bit_len
            assert frame.header_fields["payload_length"] == bit_len
            assert frame.frame_bit_length == 32 + 56 + bit_len + 16
            assert np.array_equal(frame.payload_bits, payload.payload_bits)

    def test_frame_sha256_verification(self, payload_generator, frame_generator):
        """TEST 17: frame SHA-256 verifies"""
        payload = payload_generator.generate(payload_type="random_bits", bit_length=512, seed=42)
        frame = frame_generator.generate(payload)

        computed_sha = hashlib.sha256(frame.frame_bytes).hexdigest()
        assert frame.sha256 == computed_sha

    def test_truth_parser_recovery(self, payload_generator, frame_generator):
        """TEST 20: known parser recovers original payload exactly"""
        payload = payload_generator.generate(payload_type="text", text="ASTRA GROUND TRUTH RECOVERY TEST")
        frame = frame_generator.generate(payload)

        # Parse using truth validation parser
        parsed = parse_known_frame(frame.frame_bits, frame.to_dict())

        assert parsed["crc_valid"] is True
        assert np.array_equal(parsed["sync_bits"], frame.sync_bits)
        assert np.array_equal(parsed["header_bits"], frame.header_bits)
        assert np.array_equal(parsed["payload_bits"], payload.payload_bits)
        assert np.array_equal(parsed["crc_bits"], frame.crc_bits)

    def test_continuous_stream_generation(self, payload_generator, frame_generator):
        """Test continuous multi-frame concatenation and telemetry."""
        payloads = [
            payload_generator.generate(payload_type="random_bits", bit_length=128, seed=i, payload_id=f"p_{i}")
            for i in range(5)
        ]

        stream = frame_generator.generate_stream(
            payloads,
            inter_frame_gap={"enabled": True, "length_bits": 8, "mode": "zeros"},
        )

        assert len(stream.frame_ids) == 5
        assert len(stream.frame_start_positions) == 5
        assert len(stream.gap_positions) == 4  # gaps between 5 frames

        # Verify each frame's bits in the stream
        for idx, (p_start, p_len) in enumerate(zip(stream.frame_start_positions, stream.frame_lengths)):
            extracted_frame_bits = stream.stream_bits[p_start : p_start + p_len]
            # Regenerate frame to check bit-exact equality
            expected_frame = frame_generator.generate(payloads[idx], sequence_number=idx)
            assert np.array_equal(extracted_frame_bits, expected_frame.frame_bits)
