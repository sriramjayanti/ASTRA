"""
Sanitized Visible Metadata Builder for ASTRA Dataset Orchestration (Engine 8).
Exposes only physical capture attributes to blind receivers with absolute label leakage protection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from astra_synthetic.capture.models import CaptureRecord


def build_visible_metadata_document(
    cap_rec: CaptureRecord,
    expose_sample_rate: bool = True,
    expose_center_freq: bool = True,
) -> dict[str, Any]:
    """Construct sanitized capture metadata visible to blind receivers/ingestion algorithms.

    Guarantees that no hidden ground truth (modulation class, FEC, interleaver, payload, SNR, CFO)
    is exposed.
    """
    file_p = Path(cap_rec.file_path)

    return {
        "capture_record_id": cap_rec.capture_record_id,
        "file_name": file_p.name,
        "file_format": cap_rec.file_format,
        "sample_rate_hz": float(cap_rec.sample_rate_hz) if (expose_sample_rate and cap_rec.sample_rate_visible) else None,
        "center_frequency_hz": float(cap_rec.center_frequency_hz) if (expose_center_freq and cap_rec.center_frequency_hz is not None) else None,
        "sample_count_complex": cap_rec.sample_count_complex,
        "scalar_sample_count": cap_rec.scalar_sample_count,
        "duration_seconds": round(float(cap_rec.duration_seconds), 6),
        "storage_dtype": cap_rec.storage_dtype,
        "endianness": cap_rec.endianness,
        "iq_order": cap_rec.iq_order,
        "channel_count": cap_rec.channel_count,
        "file_size_bytes": cap_rec.file_size_bytes,
        "file_sha256": cap_rec.file_sha256,
    }
