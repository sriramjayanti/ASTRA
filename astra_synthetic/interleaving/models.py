"""
Data models for ASTRA Interleaving / Bit-Reordering Engine.
Encapsulates ground-truth interleaved bitstreams, index permutation mappings, and block telemetry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class InterleaverRecord:
    """Represents a complete Interleaver Record with exact ground-truth permutation tracking.

    Attributes:
        interleaver_record_id: Unique record ID (e.g. 'int_000001').
        fec_record_id: Source FECRecord ID.
        frame_id: Source FrameRecord ID.
        interleaver_type: Interleaver family ('none', 'block', 'convolutional', 'diagonal', 'pseudo_random').
        profile_name: Name of the applied profile (e.g. 'block_16x32', 'conv_b4_d2', 'pr_256').
        input_bits: Original FEC-encoded input bits (immutable ground truth).
        interleaved_bits: Reordered output bitstream.
        input_bit_length: Bit length of input_bits.
        output_bit_length: Bit length of interleaved_bits.
        padding_bits: Explicit padding bits added for block/matrix alignment.
        padding_length: Count of padding bits added.
        parameters: Full interleaver parameter dictionary.
        permutation: 1D NumPy array of source index mappings for output positions (if finite block).
        inverse_permutation: 1D NumPy array of inverse mappings to recover input order.
        block_boundaries: Telemetry map of block offsets and padding.
        input_sha256: SHA-256 hash of packed input_bits.
        output_sha256: SHA-256 hash of packed interleaved_bits.
        metadata: Audit metadata dictionary.
    """
    interleaver_record_id: str
    fec_record_id: str
    frame_id: str
    interleaver_type: str
    profile_name: str | None
    input_bits: np.ndarray
    interleaved_bits: np.ndarray
    input_bit_length: int
    output_bit_length: int
    padding_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    padding_length: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)
    permutation: np.ndarray | None = None
    inverse_permutation: np.ndarray | None = None
    block_boundaries: list[dict[str, Any]] = field(default_factory=list)
    input_sha256: str = ""
    output_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure bit arrays are 1D np.uint8
        for attr in ("input_bits", "interleaved_bits", "padding_bits"):
            val = getattr(self, attr)
            if val is not None:
                if not isinstance(val, np.ndarray):
                    setattr(self, attr, np.asarray(val, dtype=np.uint8))
                elif val.dtype != np.uint8:
                    setattr(self, attr, val.astype(np.uint8))

        if self.permutation is not None and not isinstance(self.permutation, np.ndarray):
            self.permutation = np.asarray(self.permutation, dtype=np.int64)
        if self.inverse_permutation is not None and not isinstance(self.inverse_permutation, np.ndarray):
            self.inverse_permutation = np.asarray(self.inverse_permutation, dtype=np.int64)

        if not self.input_sha256 and len(self.input_bits) > 0:
            self.input_sha256 = hashlib.sha256(np.packbits(self.input_bits).tobytes()).hexdigest()
        elif not self.input_sha256:
            self.input_sha256 = hashlib.sha256(b"").hexdigest()

        if not self.output_sha256 and len(self.interleaved_bits) > 0:
            self.output_sha256 = hashlib.sha256(np.packbits(self.interleaved_bits).tobytes()).hexdigest()
        elif not self.output_sha256:
            self.output_sha256 = hashlib.sha256(b"").hexdigest()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, InterleaverRecord):
            return False
        return (
            self.interleaver_record_id == other.interleaver_record_id
            and self.fec_record_id == other.fec_record_id
            and self.frame_id == other.frame_id
            and self.interleaver_type == other.interleaver_type
            and self.profile_name == other.profile_name
            and np.array_equal(self.input_bits, other.input_bits)
            and np.array_equal(self.interleaved_bits, other.interleaved_bits)
            and self.input_bit_length == other.input_bit_length
            and self.output_bit_length == other.output_bit_length
            and self.padding_length == other.padding_length
            and self.input_sha256 == other.input_sha256
            and self.output_sha256 == other.output_sha256
            and self.parameters == other.parameters
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert InterleaverRecord metadata to JSON-serializable dictionary."""
        return {
            "interleaver_record_id": self.interleaver_record_id,
            "fec_record_id": self.fec_record_id,
            "frame_id": self.frame_id,
            "interleaver_type": self.interleaver_type,
            "profile_name": self.profile_name,
            "input_bit_length": self.input_bit_length,
            "output_bit_length": self.output_bit_length,
            "padding_length": self.padding_length,
            "parameters": self.parameters,
            "block_boundaries": self.block_boundaries,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "bit_order": self.metadata.get("bit_order", "msb_first"),
            "generator_version": self.metadata.get("generator_version", "1.0.0"),
            "metadata": self.metadata,
        }
