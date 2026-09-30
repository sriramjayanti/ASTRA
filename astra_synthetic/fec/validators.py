"""
Validation utilities for ASTRA FEC Engine.
Validates input-output bit integrity, rate mathematics, and executes reference decoder verification.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any
import numpy as np

from ..framing.models import FrameRecord
from ..payload.validators import ValidationError, validate_payload_bits
from .models import FECRecord
from .convolutional import ConvolutionalCode
from .reed_solomon import ReedSolomonCode
from .concatenated import ConcatenatedCode
from .ldpc import LDPCCode
from .profiles import get_fec_profile


def validate_fec_record(
    record: FECRecord,
    original_frame: FrameRecord | None = None,
) -> None:
    """Validate consistency, length bounds, hashes, and parameters of an FECRecord.

    Args:
        record: FECRecord instance.
        original_frame: Optional source FrameRecord to verify input bit preservation.

    Raises:
        ValidationError: If any integrity or consistency check fails.
    """
    if not isinstance(record, FECRecord):
        raise ValidationError(f"Expected FECRecord, got {type(record).__name__}")

    # 1. Validate bit arrays
    validate_payload_bits(record.input_bits)
    validate_payload_bits(record.encoded_bits)

    if len(record.input_bits) != record.input_bit_length:
        raise ValidationError(
            f"Input bit length mismatch: declared {record.input_bit_length} vs actual {len(record.input_bits)}"
        )
    if len(record.encoded_bits) != record.encoded_bit_length:
        raise ValidationError(
            f"Encoded bit length mismatch: declared {record.encoded_bit_length} vs actual {len(record.encoded_bits)}"
        )

    # 2. Check source frame bit preservation
    if original_frame is not None:
        if record.frame_id != original_frame.frame_id:
            raise ValidationError(
                f"frame_id mismatch: FECRecord {record.frame_id} vs FrameRecord {original_frame.frame_id}"
            )
        if not np.array_equal(record.input_bits, original_frame.frame_bits):
            raise ValidationError("FECRecord.input_bits do not match original FrameRecord.frame_bits!")

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

    expected_enc_sha = (
        hashlib.sha256(np.packbits(record.encoded_bits).tobytes()).hexdigest()
        if len(record.encoded_bits) > 0
        else hashlib.sha256(b"").hexdigest()
    )
    if record.encoded_sha256 != expected_enc_sha:
        raise ValidationError(
            f"Encoded SHA-256 mismatch: recorded {record.encoded_sha256} vs computed {expected_enc_sha}"
        )

    # 4. Check code rates
    if not (0.0 < record.effective_code_rate <= 1.05):
        raise ValidationError(f"Effective code rate out of reasonable range: {record.effective_code_rate}")

    # 5. Metadata JSON serializability
    try:
        json.dumps(record.to_dict())
    except Exception as e:
        raise ValidationError(f"FEC metadata is not JSON serializable: {e}") from e


def reference_decode(record: FECRecord) -> np.ndarray:
    """Execute the reference decoder for a given FECRecord to verify round-trip recoverability.

    Args:
        record: FECRecord instance.

    Returns:
        1D NumPy uint8 array of recovered bits.

    Raises:
        ValueError: If decoding fails.
    """
    fec_type = record.fec_type.lower()

    if fec_type == "none":
        return record.encoded_bits.copy()

    elif fec_type == "convolutional":
        params = record.parameters
        code = ConvolutionalCode(
            constraint_length=params["constraint_length"],
            generators=params["generators_octal"],
            generator_format="octal",
            rate=params.get("code_rate", "1/2"),
            termination_mode=params.get("termination_mode", "zero_tail"),
        )
        return code.decode_viterbi(
            encoded_bits=record.encoded_bits,
            original_bit_length=record.input_bit_length,
            termination=params.get("termination_mode", "zero_tail"),
        )

    elif fec_type == "reed_solomon":
        params = record.parameters
        code = ReedSolomonCode(
            n=params["n"],
            k=params["k"],
            symbol_size_bits=params.get("symbol_size_bits", 8),
        )
        return code.decode(
            encoded_bits=record.encoded_bits,
            original_bit_length=record.input_bit_length,
            block_boundaries=record.block_boundaries,
        )

    elif fec_type == "concatenated":
        outer_p = record.parameters["outer"]
        inner_p = record.parameters["inner"]
        outer_rs = ReedSolomonCode(
            n=outer_p["n"],
            k=outer_p["k"],
            symbol_size_bits=outer_p.get("symbol_size_bits", 8),
        )
        inner_conv = ConvolutionalCode(
            constraint_length=inner_p["constraint_length"],
            generators=inner_p["generators_octal"],
            generator_format="octal",
            rate=inner_p.get("code_rate", "1/2"),
            termination_mode=inner_p.get("termination_mode", "zero_tail"),
        )
        concat = ConcatenatedCode(outer_rs=outer_rs, inner_conv=inner_conv)
        
        rs_len = len(record.intermediate_stages.get("rs_encoded_bits", np.empty(0)))
        if rs_len == 0:
            # Reconstruct expected RS bit length from block count
            rs_len = record.block_count * outer_rs.n * 8

        return concat.decode(
            encoded_bits=record.encoded_bits,
            original_bit_length=record.input_bit_length,
            intermediate_rs_length=rs_len,
            block_boundaries=record.block_boundaries,
        )

    elif fec_type == "ldpc":
        params = record.parameters
        prof_id = params.get("profile_id", record.fec_profile)
        ldpc = LDPCCode(profile=prof_id)
        return ldpc.decode_reference(
            encoded_bits=record.encoded_bits,
            original_bit_length=record.input_bit_length,
            block_boundaries=record.block_boundaries,
        )

    else:
        raise ValueError(f"Unknown FEC type '{fec_type}' for reference decoding.")
