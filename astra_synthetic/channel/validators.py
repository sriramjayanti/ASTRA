"""
Validation utilities for ASTRA Channel Engine (Engine 6).
Verifies IQ finite properties, SHA-256 integrity, source modulation fidelity, and metadata validity.
"""

from __future__ import annotations

import json
import hashlib
import numpy as np

from ..modulation.models import ModulationRecord
from ..payload.validators import ValidationError
from .models import ChannelRecord


def validate_channel_record(
    record: ChannelRecord,
    original_modulation: ModulationRecord | None = None,
) -> None:
    """Validate completeness, mathematical properties, and SHA-256 integrity of a ChannelRecord.

    Args:
        record: ChannelRecord instance.
        original_modulation: Optional source ModulationRecord to verify clean IQ preservation.

    Raises:
        ValidationError: If any consistency or integrity check fails.
    """
    if not isinstance(record, ChannelRecord):
        raise ValidationError(f"Expected ChannelRecord, got {type(record).__name__}")

    # 1. Validate clean IQ and impaired IQ arrays
    if record.clean_iq.dtype != np.complex64:
        raise ValidationError(f"clean_iq must have dtype complex64, got {record.clean_iq.dtype}")
    if record.impaired_iq.dtype != np.complex64:
        raise ValidationError(f"impaired_iq must have dtype complex64, got {record.impaired_iq.dtype}")

    if not np.all(np.isfinite(record.clean_iq)):
        raise ValidationError("clean_iq contains non-finite values (NaN or Inf)!")
    if not np.all(np.isfinite(record.impaired_iq)):
        raise ValidationError("impaired_iq contains non-finite values (NaN or Inf)!")

    # 2. Check clean IQ preservation from source ModulationRecord
    if original_modulation is not None:
        if record.modulation_record_id != original_modulation.modulation_record_id:
            raise ValidationError(
                f"modulation_record_id mismatch: {record.modulation_record_id} vs {original_modulation.modulation_record_id}"
            )
        if not np.array_equal(record.clean_iq, original_modulation.clean_iq):
            raise ValidationError("ChannelRecord.clean_iq does not match source ModulationRecord.clean_iq!")

    # 3. Check SHA-256 hashes
    expected_clean_sha = (
        hashlib.sha256(record.clean_iq.tobytes()).hexdigest()
        if len(record.clean_iq) > 0
        else hashlib.sha256(b"").hexdigest()
    )
    if record.clean_iq_sha256 != expected_clean_sha:
        raise ValidationError(
            f"clean_iq_sha256 mismatch: recorded {record.clean_iq_sha256} vs computed {expected_clean_sha}"
        )

    expected_impaired_sha = (
        hashlib.sha256(record.impaired_iq.tobytes()).hexdigest()
        if len(record.impaired_iq) > 0
        else hashlib.sha256(b"").hexdigest()
    )
    if record.impaired_iq_sha256 != expected_impaired_sha:
        raise ValidationError(
            f"impaired_iq_sha256 mismatch: recorded {record.impaired_iq_sha256} vs computed {expected_impaired_sha}"
        )

    # 4. Validate sample rate
    if record.sample_rate <= 0:
        raise ValidationError(f"sample_rate must be > 0, got {record.sample_rate}")

    # 5. Metadata JSON serializability
    try:
        json.dumps(record.to_dict())
    except Exception as e:
        raise ValidationError(f"Channel metadata is not JSON serializable: {e}") from e
