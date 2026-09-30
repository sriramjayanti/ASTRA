"""
Data models for ASTRA Framing Engine.
Encapsulates complete framed bitstreams and ground-truth boundary telemetry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class FrameRecord:
    """Represents a complete structured communication frame with ground truth region tracking.

    Attributes:
        frame_id: Unique identifier (e.g. 'frame_000001').
        payload_id: ID of the underlying source PayloadRecord.
        frame_bits: Complete framed bitstream [SYNC | HEADER | PAYLOAD | CRC].
        frame_bytes: Packed frame byte representation.
        frame_bit_length: Total number of bits in the frame.
        sync_bits: Synchronization word bit array.
        header_bits: Structured header bit array.
        payload_bits: Original source payload bit array.
        crc_bits: Integrity check bit array.
        sync_start: Start bit index of SYNC region.
        sync_length: Bit length of SYNC region.
        header_start: Start bit index of HEADER region.
        header_length: Bit length of HEADER region.
        payload_start: Start bit index of PAYLOAD region.
        payload_length: Bit length of PAYLOAD region.
        crc_start: Start bit index of CRC region.
        crc_length: Bit length of CRC region.
        sync_type: Generation mode of sync word ('fixed', 'pool', 'random').
        sync_value: String representation of sync word (hex or bitstring).
        header_schema: Name of header schema applied.
        header_fields: Dictionary of decoded header field values.
        header_field_offsets: Field-level ground truth offsets within header.
        crc_type: Name of CRC algorithm profile used.
        crc_value: Integer or hexadecimal value of the CRC.
        crc_scope: Regions covered by CRC (e.g. ['header', 'payload']).
        frame_sequence_number: Sequence number of this frame.
        metadata: Complete audit metadata dictionary.
        sha256: Hexadecimal SHA-256 hash of packed frame_bytes.
    """
    frame_id: str
    payload_id: str
    frame_bits: np.ndarray
    frame_bytes: bytes
    frame_bit_length: int
    sync_bits: np.ndarray
    header_bits: np.ndarray
    payload_bits: np.ndarray
    crc_bits: np.ndarray
    sync_start: int
    sync_length: int
    header_start: int
    header_length: int
    payload_start: int
    payload_length: int
    crc_start: int
    crc_length: int
    sync_type: str
    sync_value: str
    header_schema: str
    header_fields: dict[str, Any]
    header_field_offsets: dict[str, dict[str, int]]
    crc_type: str
    crc_value: int | str
    crc_scope: list[str]
    frame_sequence_number: int
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""

    def __post_init__(self):
        # Ensure bit arrays are 1D np.uint8
        for attr in ("frame_bits", "sync_bits", "header_bits", "payload_bits", "crc_bits"):
            val = getattr(self, attr)
            if not isinstance(val, np.ndarray):
                setattr(self, attr, np.asarray(val, dtype=np.uint8))
            elif val.dtype != np.uint8:
                setattr(self, attr, val.astype(np.uint8))
                
        if not self.sha256 and self.frame_bytes is not None:
            self.sha256 = hashlib.sha256(self.frame_bytes).hexdigest()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrameRecord):
            return False
        return (
            self.frame_id == other.frame_id
            and self.payload_id == other.payload_id
            and np.array_equal(self.frame_bits, other.frame_bits)
            and self.frame_bytes == other.frame_bytes
            and self.frame_bit_length == other.frame_bit_length
            and np.array_equal(self.sync_bits, other.sync_bits)
            and np.array_equal(self.header_bits, other.header_bits)
            and np.array_equal(self.payload_bits, other.payload_bits)
            and np.array_equal(self.crc_bits, other.crc_bits)
            and self.sync_start == other.sync_start
            and self.sync_length == other.sync_length
            and self.header_start == other.header_start
            and self.header_length == other.header_length
            and self.payload_start == other.payload_start
            and self.payload_length == other.payload_length
            and self.crc_start == other.crc_start
            and self.crc_length == other.crc_length
            and self.sync_type == other.sync_type
            and self.sync_value == other.sync_value
            and self.header_schema == other.header_schema
            and self.header_fields == other.header_fields
            and self.crc_type == other.crc_type
            and self.crc_value == other.crc_value
            and self.crc_scope == other.crc_scope
            and self.frame_sequence_number == other.frame_sequence_number
            and self.sha256 == other.sha256
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert frame structure and ground-truth metadata to JSON-serializable dictionary."""
        return {
            "frame_id": self.frame_id,
            "payload_id": self.payload_id,
            "layout": ["sync", "header", "payload", "crc"],
            "frame_bit_length": self.frame_bit_length,
            "frame_sequence_number": self.frame_sequence_number,
            "sync": {
                "start": self.sync_start,
                "length": self.sync_length,
                "type": self.sync_type,
                "value": self.sync_value,
            },
            "header": {
                "start": self.header_start,
                "length": self.header_length,
                "schema": self.header_schema,
                "fields": self.header_fields,
                "field_offsets": self.header_field_offsets,
            },
            "payload": {
                "start": self.payload_start,
                "length": self.payload_length,
            },
            "crc": {
                "start": self.crc_start,
                "length": self.crc_length,
                "type": self.crc_type,
                "value": str(self.crc_value),
                "scope": self.crc_scope,
            },
            "bit_order": self.metadata.get("bit_order", "msb_first"),
            "sha256": self.sha256,
            "generator_version": self.metadata.get("generator_version", "1.0.0"),
            "metadata": self.metadata,
        }


@dataclass
class FrameStreamRecord:
    """Represents a continuous concatenated sequence of multiple frames for correlation analysis."""
    stream_bits: np.ndarray
    stream_bytes: bytes
    total_bit_length: int
    frame_ids: list[str]
    frame_start_positions: list[int]
    frame_lengths: list[int]
    sync_positions: list[dict[str, int]]
    header_positions: list[dict[str, int]]
    payload_positions: list[dict[str, int]]
    crc_positions: list[dict[str, int]]
    gap_positions: list[dict[str, int]]
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""

    def __post_init__(self):
        if not isinstance(self.stream_bits, np.ndarray):
            self.stream_bits = np.asarray(self.stream_bits, dtype=np.uint8)
        if not self.sha256 and self.stream_bytes is not None:
            self.sha256 = hashlib.sha256(self.stream_bytes).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert stream telemetry to JSON-serializable dictionary."""
        return {
            "total_bit_length": self.total_bit_length,
            "total_frames": len(self.frame_ids),
            "frame_ids": self.frame_ids,
            "frame_start_positions": self.frame_start_positions,
            "frame_lengths": self.frame_lengths,
            "sync_positions": self.sync_positions,
            "header_positions": self.header_positions,
            "payload_positions": self.payload_positions,
            "crc_positions": self.crc_positions,
            "gap_positions": self.gap_positions,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }
