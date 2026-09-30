"""
Validation utilities for ASTRA Framing Engine.
Validates boundary integrity, bit consistency, CRC correctness, and provides a truth validation parser.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any
import numpy as np

from ..payload.models import PayloadRecord
from ..payload.validators import ValidationError, validate_payload_bits
from .models import FrameRecord
from .crc import compute_crc, verify_crc, get_crc_profile


def validate_frame_record(
    frame: FrameRecord,
    original_payload: PayloadRecord | None = None,
) -> None:
    """Validate completeness, region boundaries, CRC, and ground-truth preservation of a FrameRecord.

    Args:
        frame: FrameRecord instance.
        original_payload: Optional original PayloadRecord to verify exact bit preservation.

    Raises:
        ValidationError: If any consistency or integrity check fails.
    """
    if not isinstance(frame, FrameRecord):
        raise ValidationError(f"Expected FrameRecord, got {type(frame).__name__}")

    # 1. Validate bit array properties
    validate_payload_bits(frame.frame_bits)
    if len(frame.frame_bits) != frame.frame_bit_length:
        raise ValidationError(
            f"Frame bit length mismatch: declared {frame.frame_bit_length} vs actual {len(frame.frame_bits)}"
        )

    # 2. Validate region boundaries and non-overlap
    if frame.sync_start != 0:
        raise ValidationError(f"sync_start must be 0, got {frame.sync_start}")
    
    if frame.header_start != frame.sync_start + frame.sync_length:
        raise ValidationError(
            f"header_start ({frame.header_start}) must equal sync_start + sync_length ({frame.sync_start + frame.sync_length})"
        )
        
    if frame.payload_start != frame.header_start + frame.header_length:
        raise ValidationError(
            f"payload_start ({frame.payload_start}) must equal header_start + header_length ({frame.header_start + frame.header_length})"
        )
        
    if frame.crc_start != frame.payload_start + frame.payload_length:
        raise ValidationError(
            f"crc_start ({frame.crc_start}) must equal payload_start + payload_length ({frame.payload_start + frame.payload_length})"
        )
        
    if frame.crc_start + frame.crc_length != frame.frame_bit_length:
        raise ValidationError(
            f"End of CRC region ({frame.crc_start + frame.crc_length}) does not match frame_bit_length ({frame.frame_bit_length})"
        )

    # 3. Validate slice equality
    extracted_sync = frame.frame_bits[frame.sync_start : frame.sync_start + frame.sync_length]
    if not np.array_equal(extracted_sync, frame.sync_bits):
        raise ValidationError("Slices in frame_bits do not match sync_bits")

    extracted_header = frame.frame_bits[frame.header_start : frame.header_start + frame.header_length]
    if not np.array_equal(extracted_header, frame.header_bits):
        raise ValidationError("Slices in frame_bits do not match header_bits")

    extracted_payload = frame.frame_bits[frame.payload_start : frame.payload_start + frame.payload_length]
    if not np.array_equal(extracted_payload, frame.payload_bits):
        raise ValidationError("Slices in frame_bits do not match payload_bits")

    extracted_crc = frame.frame_bits[frame.crc_start : frame.crc_start + frame.crc_length]
    if not np.array_equal(extracted_crc, frame.crc_bits):
        raise ValidationError("Slices in frame_bits do not match crc_bits")

    # 4. Check payload integrity against original PayloadRecord if provided
    if original_payload is not None:
        if not np.array_equal(frame.payload_bits, original_payload.payload_bits):
            raise ValidationError(
                f"Framed payload bits do not match original source payload '{original_payload.payload_id}'"
            )

    # 5. Validate CRC recomputation over designated scope
    crc_scope_bits: list[np.ndarray] = []
    for region in frame.crc_scope:
        if region == "header":
            crc_scope_bits.append(frame.header_bits)
        elif region == "payload":
            crc_scope_bits.append(frame.payload_bits)
        elif region == "sync":
            crc_scope_bits.append(frame.sync_bits)
        else:
            raise ValidationError(f"Unknown CRC scope region '{region}'")

    if crc_scope_bits:
        crc_input = np.concatenate(crc_scope_bits)
        if not verify_crc(crc_input, frame.crc_bits, crc_type=frame.crc_type):
            raise ValidationError(
                f"CRC verification failed! Computed CRC does not match stored CRC for type '{frame.crc_type}'"
            )

    # 6. Validate SHA-256 checksum
    expected_sha = hashlib.sha256(frame.frame_bytes).hexdigest()
    if frame.sha256 != expected_sha:
        raise ValidationError(
            f"Frame SHA-256 mismatch: recorded {frame.sha256} vs computed {expected_sha}"
        )

    # 7. Validate metadata JSON serializability
    try:
        json.dumps(frame.to_dict())
    except Exception as e:
        raise ValidationError(f"Frame metadata is not JSON serializable: {e}") from e


def parse_known_frame(
    frame_bits: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Truth validation parser: Extracts frame regions according to known metadata ground truth.

    Used to verify that framing generation, ground-truth offset indexing, and payload preservation
    are 100% consistent and recoverable.

    Args:
        frame_bits: 1D NumPy array of frame bits.
        metadata: Metadata dictionary or FrameRecord.to_dict() output.

    Returns:
        Dictionary containing extracted 'sync_bits', 'header_bits', 'payload_bits', 'crc_bits',
        'crc_valid', and decoded 'header_fields'.
    """
    if not isinstance(frame_bits, np.ndarray):
        frame_bits = np.asarray(frame_bits, dtype=np.uint8)

    sync_info = metadata["sync"]
    header_info = metadata["header"]
    payload_info = metadata["payload"]
    crc_info = metadata["crc"]

    sync_start = sync_info["start"]
    sync_len = sync_info["length"]

    header_start = header_info["start"]
    header_len = header_info["length"]

    payload_start = payload_info["start"]
    payload_len = payload_info["length"]

    crc_start = crc_info["start"]
    crc_len = crc_info["length"]

    # Extract slices
    sync_bits = frame_bits[sync_start : sync_start + sync_len]
    header_bits = frame_bits[header_start : header_start + header_len]
    payload_bits = frame_bits[payload_start : payload_start + payload_len]
    crc_bits = frame_bits[crc_start : crc_start + crc_len]

    # Reconstruct CRC scope and test validity
    scope_chunks = []
    for reg in crc_info.get("scope", ["header", "payload"]):
        if reg == "header":
            scope_chunks.append(header_bits)
        elif reg == "payload":
            scope_chunks.append(payload_bits)
        elif reg == "sync":
            scope_chunks.append(sync_bits)

    crc_valid = False
    if scope_chunks:
        crc_input = np.concatenate(scope_chunks)
        crc_valid = verify_crc(crc_input, crc_bits, crc_type=crc_info["type"])

    return {
        "sync_bits": sync_bits,
        "header_bits": header_bits,
        "payload_bits": payload_bits,
        "crc_bits": crc_bits,
        "crc_valid": crc_valid,
        "header_fields": header_info.get("fields", {}),
        "payload_length": len(payload_bits),
    }
