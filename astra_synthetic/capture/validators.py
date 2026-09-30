"""
Validation suite for ASTRA Capture Generator (Engine 7).
Verifies physical file consistency, expected file byte size, non-mutation of input ChannelRecord,
and visible/truth metadata separation (leak prevention).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from astra_synthetic.payload.validators import ValidationError
from astra_synthetic.channel.models import ChannelRecord
from .models import CaptureRecord


DTYPE_BYTES = {
    "float32": 4,
    "f32": 4,
    "float64": 8,
    "f64": 8,
    "int16": 2,
    "i16": 2,
    "int8": 1,
    "i8": 1,
}


def validate_capture_record(
    record: CaptureRecord,
    original_channel: ChannelRecord | None = None,
) -> None:
    """Validate completeness, mathematical properties, and physical file integrity of a CaptureRecord.

    Args:
        record: CaptureRecord instance.
        original_channel: Optional source ChannelRecord to verify non-mutation.

    Raises:
        ValidationError: If any physical or logical validation check fails.
    """
    if not isinstance(record, CaptureRecord):
        raise ValidationError(f"Expected CaptureRecord, got {type(record).__name__}")

    # 1. File existence
    file_path = Path(record.file_path)
    if not file_path.is_file():
        raise ValidationError(f"Capture file does not exist: {file_path}")

    # 2. File size verification
    actual_size = file_path.stat().st_size
    bytes_per_scalar = DTYPE_BYTES.get(record.storage_dtype.lower(), 4)
    expected_payload_bytes = record.sample_count_complex * 2 * bytes_per_scalar

    if record.file_format == "raw_iq":
        if actual_size != expected_payload_bytes:
            raise ValidationError(
                f"Raw IQ file size mismatch: expected {expected_payload_bytes} bytes, found {actual_size} bytes."
            )
    elif record.file_format == "wav":
        # WAV has RIFF header (~44 bytes standard PCM, or 44+ bytes for extended formats)
        if actual_size < expected_payload_bytes + 44:
            raise ValidationError(
                f"WAV file size too small: expected >= {expected_payload_bytes + 44} bytes, found {actual_size} bytes."
            )
    elif record.file_format == "sigmf":
        if actual_size != expected_payload_bytes:
            raise ValidationError(
                f"SigMF data file size mismatch: expected {expected_payload_bytes} bytes, found {actual_size} bytes."
            )
        meta_file = file_path.parent / f"{file_path.stem}.sigmf-meta"
        if not meta_file.is_file():
            raise ValidationError(f"SigMF metadata file missing: {meta_file}")

    # 3. File SHA-256 hash check
    with open(file_path, "rb") as f:
        computed_sha = hashlib.sha256(f.read()).hexdigest()
    if record.file_sha256 != computed_sha:
        raise ValidationError(
            f"File SHA-256 mismatch: recorded {record.file_sha256} vs computed {computed_sha}"
        )

    # 4. Source ChannelRecord immutability & SHA check
    if original_channel is not None:
        if record.channel_record_id != original_channel.channel_record_id:
            raise ValidationError(
                f"channel_record_id mismatch: {record.channel_record_id} vs {original_channel.channel_record_id}"
            )
        expected_src_sha = hashlib.sha256(original_channel.impaired_iq.tobytes()).hexdigest()
        if record.source_iq_sha256 != expected_src_sha:
            raise ValidationError("CaptureRecord source_iq_sha256 does not match original ChannelRecord IQ!")

    # 5. Numerical and metadata properties
    if record.sample_rate_hz <= 0:
        raise ValidationError(f"Sample rate must be > 0, got {record.sample_rate_hz}")

    expected_duration = record.sample_count_complex / record.sample_rate_hz
    if not np.isclose(record.duration_seconds, expected_duration, atol=1e-5):
        raise ValidationError(
            f"Duration mismatch: recorded {record.duration_seconds}s vs expected {expected_duration}s"
        )

    # 6. Leak prevention check: Visible metadata must NOT contain hidden ground truth labels
    visible_dict = record.to_visible_dict()
    forbidden_keys = [
        "modulation",
        "modulation_type",
        "fec",
        "interleaver",
        "snr_db",
        "cfo_hz",
        "payload_bits",
    ]
    for k in forbidden_keys:
        if k in visible_dict:
            raise ValidationError(f"CRITICAL LEAKAGE: Forbidden key '{k}' found in visible metadata!")
