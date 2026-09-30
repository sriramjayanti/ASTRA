"""
Data models for ASTRA FEC (Forward Error Correction) Engine.
Encapsulates ground-truth FEC-encoded bitstreams, parameter metadata, and block telemetry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class FECRecord:
    """Represents a complete FEC-encoded record with comprehensive ground-truth metadata.

    Attributes:
        fec_record_id: Unique identifier (e.g. 'fec_000001').
        frame_id: Source FrameRecord ID.
        fec_type: FEC family name ('none', 'convolutional', 'reed_solomon', 'concatenated', 'ldpc').
        fec_profile: Specific profile name (e.g. 'conv_k7_r12', 'rs_255_223', 'ldpc_n128_k64_r12').
        input_bits: Original source frame bits (immutable ground truth).
        encoded_bits: Output FEC-encoded bitstream.
        input_bit_length: Bit count of input_bits.
        encoded_bit_length: Bit count of encoded_bits.
        nominal_code_rate: Mathematical design code rate k/n (e.g. 0.5, 0.8745, 1.0).
        effective_code_rate: True transmission rate = input_bit_length / encoded_bit_length.
        padding_bits: Any padding bits added for byte alignment or block segmentation.
        padding_length: Number of padding bits added.
        block_count: Number of codewords/blocks in this record.
        block_boundaries: Detailed offset mapping of each block (start, length, padding).
        parameters: Full coding parameters (polynomials, constraint length, n, k, etc.).
        metadata: Audit metadata and generator version.
        input_sha256: SHA-256 of packed input_bits.
        encoded_sha256: SHA-256 of packed encoded_bits.
        intermediate_stages: Optional intermediate stage bit arrays (e.g. for concatenated codes).
    """
    fec_record_id: str
    frame_id: str
    fec_type: str
    fec_profile: str | None
    input_bits: np.ndarray
    encoded_bits: np.ndarray
    input_bit_length: int
    encoded_bit_length: int
    nominal_code_rate: float
    effective_code_rate: float
    padding_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    padding_length: int = 0
    block_count: int = 1
    block_boundaries: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    input_sha256: str = ""
    encoded_sha256: str = ""
    intermediate_stages: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure bit arrays are 1D np.uint8
        for attr in ("input_bits", "encoded_bits", "padding_bits"):
            val = getattr(self, attr)
            if not isinstance(val, np.ndarray):
                setattr(self, attr, np.asarray(val, dtype=np.uint8))
            elif val.dtype != np.uint8:
                setattr(self, attr, val.astype(np.uint8))

        if not self.input_sha256 and len(self.input_bits) > 0:
            self.input_sha256 = hashlib.sha256(np.packbits(self.input_bits).tobytes()).hexdigest()
        elif not self.input_sha256:
            self.input_sha256 = hashlib.sha256(b"").hexdigest()

        if not self.encoded_sha256 and len(self.encoded_bits) > 0:
            self.encoded_sha256 = hashlib.sha256(np.packbits(self.encoded_bits).tobytes()).hexdigest()
        elif not self.encoded_sha256:
            self.encoded_sha256 = hashlib.sha256(b"").hexdigest()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FECRecord):
            return False
        return (
            self.fec_record_id == other.fec_record_id
            and self.frame_id == other.frame_id
            and self.fec_type == other.fec_type
            and self.fec_profile == other.fec_profile
            and np.array_equal(self.input_bits, other.input_bits)
            and np.array_equal(self.encoded_bits, other.encoded_bits)
            and self.input_bit_length == other.input_bit_length
            and self.encoded_bit_length == other.encoded_bit_length
            and abs(self.nominal_code_rate - other.nominal_code_rate) < 1e-5
            and abs(self.effective_code_rate - other.effective_code_rate) < 1e-5
            and self.padding_length == other.padding_length
            and self.block_count == other.block_count
            and self.input_sha256 == other.input_sha256
            and self.encoded_sha256 == other.encoded_sha256
            and self.parameters == other.parameters
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert FECRecord to JSON-serializable dictionary."""
        return {
            "fec_record_id": self.fec_record_id,
            "frame_id": self.frame_id,
            "fec_type": self.fec_type,
            "fec_profile": self.fec_profile,
            "input_bit_length": self.input_bit_length,
            "encoded_bit_length": self.encoded_bit_length,
            "nominal_code_rate": round(self.nominal_code_rate, 6),
            "effective_code_rate": round(self.effective_code_rate, 6),
            "padding_length": self.padding_length,
            "block_count": self.block_count,
            "block_boundaries": self.block_boundaries,
            "parameters": self.parameters,
            "input_sha256": self.input_sha256,
            "encoded_sha256": self.encoded_sha256,
            "bit_order": self.metadata.get("bit_order", "msb_first"),
            "generator_version": self.metadata.get("generator_version", "1.0.0"),
            "metadata": self.metadata,
        }
