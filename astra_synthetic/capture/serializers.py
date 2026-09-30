"""
Serialization and Metadata Management for ASTRA Capture Engine (Engine 7).
Writes separated visible capture metadata, hidden truth metadata, and dataset master manifests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

from .models import CaptureRecord
from .validators import validate_capture_record


def save_capture_metadata_pair(
    record: CaptureRecord,
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write paired visible (.capture.json) and truth (.truth.json) metadata files.

    Args:
        record: CaptureRecord instance.
        output_dir: Target directory (defaults to file_path parent).

    Returns:
        tuple (visible_meta_path, truth_meta_path)
    """
    file_path = Path(record.file_path)
    base_dir = Path(output_dir) if output_dir else file_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    stem = file_path.stem
    if stem.endswith(".sigmf"):
        stem = stem[:-6]

    visible_path = base_dir / f"{stem}.capture.json"
    truth_path = base_dir / f"{stem}.truth.json"

    # Save visible metadata
    with open(visible_path, "w", encoding="utf-8") as f:
        json.dump(record.to_visible_dict(), f, indent=2)

    # Save hidden truth metadata
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(record.to_truth_dict(), f, indent=2)

    record.visible_metadata_path = str(visible_path)
    record.truth_metadata_path = str(truth_path)

    return visible_path, truth_path


def write_dataset_manifest(
    records: Sequence[CaptureRecord],
    output_csv_path: Path | str,
) -> Path:
    """Generate a master CSV manifest summarizing the capture dataset for evaluation/training.

    Args:
        records: Sequence of CaptureRecord instances.
        output_csv_path: Target CSV file path.

    Returns:
        Path to the written CSV file.
    """
    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "capture_record_id",
        "channel_record_id",
        "modulation_record_id",
        "interleaver_record_id",
        "fec_record_id",
        "frame_id",
        "file_path",
        "file_format",
        "storage_dtype",
        "endianness",
        "iq_order",
        "sample_rate_hz_truth",
        "sample_rate_visible",
        "sample_count_complex",
        "duration_seconds",
        "quantization_enabled",
        "quantization_bits",
        "quantization_scale",
        "clipping_enabled",
        "file_sha256",
        "visible_metadata_path",
        "truth_metadata_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "capture_record_id": r.capture_record_id,
                "channel_record_id": r.channel_record_id,
                "modulation_record_id": r.modulation_record_id,
                "interleaver_record_id": r.interleaver_record_id,
                "fec_record_id": r.fec_record_id,
                "frame_id": r.frame_id,
                "file_path": r.file_path,
                "file_format": r.file_format,
                "storage_dtype": r.storage_dtype,
                "endianness": r.endianness,
                "iq_order": r.iq_order,
                "sample_rate_hz_truth": r.sample_rate_hz,
                "sample_rate_visible": r.sample_rate_visible,
                "sample_count_complex": r.sample_count_complex,
                "duration_seconds": r.duration_seconds,
                "quantization_enabled": r.quantization_enabled,
                "quantization_bits": r.quantization_bits,
                "quantization_scale": r.quantization_scale,
                "clipping_enabled": r.clipping_enabled,
                "file_sha256": r.file_sha256,
                "visible_metadata_path": r.visible_metadata_path,
                "truth_metadata_path": r.truth_metadata_path,
            })

    return csv_path
