"""
Validation and reference deinterleaving utilities for ASTRA Interleaving Engine.
Verifies bitstream integrity, permutation validity, and executes bit-exact round-trip recovery checks.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any
import numpy as np

from ..fec.models import FECRecord
from ..payload.validators import ValidationError, validate_payload_bits
from .models import InterleaverRecord
from .permutation import validate_permutation
from .block import BlockInterleaver
from .convolutional import ConvolutionalInterleaver
from .diagonal import DiagonalInterleaver
from .pseudo_random import PseudoRandomInterleaver


def validate_interleaver_record(
    record: InterleaverRecord,
    original_fec: FECRecord | None = None,
) -> None:
    """Validate completeness, length equations, hashes, and permutation properties of an InterleaverRecord.

    Args:
        record: InterleaverRecord instance.
        original_fec: Optional source FECRecord to verify input bit preservation.

    Raises:
        ValidationError: If any consistency or integrity check fails.
    """
    if not isinstance(record, InterleaverRecord):
        raise ValidationError(f"Expected InterleaverRecord, got {type(record).__name__}")

    # 1. Validate bit arrays
    validate_payload_bits(record.input_bits)
    validate_payload_bits(record.interleaved_bits)

    if len(record.input_bits) != record.input_bit_length:
        raise ValidationError(
            f"Input bit length mismatch: declared {record.input_bit_length} vs actual {len(record.input_bits)}"
        )
    if len(record.interleaved_bits) != record.output_bit_length:
        raise ValidationError(
            f"Output bit length mismatch: declared {record.output_bit_length} vs actual {len(record.interleaved_bits)}"
        )

    # 2. Check source FEC bit preservation
    if original_fec is not None:
        if record.fec_record_id != original_fec.fec_record_id:
            raise ValidationError(
                f"fec_record_id mismatch: {record.fec_record_id} vs {original_fec.fec_record_id}"
            )
        if not np.array_equal(record.input_bits, original_fec.encoded_bits):
            raise ValidationError("InterleaverRecord.input_bits do not match source FECRecord.encoded_bits!")

    # 3. Check hashes
    expected_in_sha = (
        hashlib.sha256(np.packbits(record.input_bits).tobytes()).hexdigest()
        if len(record.input_bits) > 0
        else hashlib.sha256(b"").hexdigest()
    )
    if record.input_sha256 != expected_in_sha:
        raise ValidationError(
            f"Input SHA-256 mismatch: recorded {record.input_sha256} vs computed {expected_in_sha}"
        )

    expected_out_sha = (
        hashlib.sha256(np.packbits(record.interleaved_bits).tobytes()).hexdigest()
        if len(record.interleaved_bits) > 0
        else hashlib.sha256(b"").hexdigest()
    )
    if record.output_sha256 != expected_out_sha:
        raise ValidationError(
            f"Output SHA-256 mismatch: recorded {record.output_sha256} vs computed {expected_out_sha}"
        )

    # 4. Check permutation array if present
    if record.permutation is not None:
        validate_permutation(record.permutation)

    # 5. Metadata JSON serializability
    try:
        json.dumps(record.to_dict())
    except Exception as e:
        raise ValidationError(f"Interleaver metadata is not JSON serializable: {e}") from e


def reference_deinterleave(record: InterleaverRecord) -> np.ndarray:
    """Execute the reference deinterleaver for an InterleaverRecord to verify exact bit recovery.

    Args:
        record: InterleaverRecord instance.

    Returns:
        1D NumPy uint8 array of recovered bits.

    Raises:
        ValueError: If deinterleaving fails.
    """
    itype = record.interleaver_type.lower()
    params = record.parameters

    if itype == "none":
        return record.interleaved_bits.copy()

    elif itype == "block":
        interleaver = BlockInterleaver(
            rows=params["rows"],
            columns=params["columns"],
            write_order=params.get("write_order", "row_major"),
            read_order=params.get("read_order", "column_major"),
            pad_mode=params.get("pad_mode", "zeros"),
        )
        return interleaver.deinterleave(
            interleaved_bits=record.interleaved_bits,
            original_bit_length=record.input_bit_length,
            block_boundaries=record.block_boundaries,
        )

    elif itype == "convolutional":
        interleaver = ConvolutionalInterleaver(
            num_branches=params["num_branches"],
            delay_step=params["delay_step"],
        )
        return interleaver.deinterleave(
            interleaved_bits=record.interleaved_bits,
            original_bit_length=record.input_bit_length,
            block_boundaries=record.block_boundaries,
        )

    elif itype == "diagonal":
        interleaver = DiagonalInterleaver(
            rows=params["rows"],
            columns=params["columns"],
            direction=params.get("direction", "top_left_to_bottom_right"),
            pad_mode=params.get("pad_mode", "zeros"),
        )
        return interleaver.deinterleave(
            interleaved_bits=record.interleaved_bits,
            original_bit_length=record.input_bit_length,
            block_boundaries=record.block_boundaries,
        )

    elif itype == "pseudo_random":
        interleaver = PseudoRandomInterleaver(
            block_size=params["block_size"],
            seed=params.get("seed", 42),
            pad_mode=params.get("pad_mode", "zeros"),
        )
        return interleaver.deinterleave(
            interleaved_bits=record.interleaved_bits,
            original_bit_length=record.input_bit_length,
            block_boundaries=record.block_boundaries,
        )

    else:
        raise ValueError(f"Unknown interleaver type '{itype}' for reference deinterleaving.")
