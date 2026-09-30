"""
Serialization utilities for ASTRA FEC Engine.
Persists FEC-encoded records to disk with full metadata and bit arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import numpy as np

from .models import FECRecord
from .validators import validate_fec_record


def save_fec_record(
    record: FECRecord,
    output_dir: Path | str,
) -> Path:
    """Save an individual FECRecord to disk.

    Creates `<output_dir>/<fec_record_id>/` containing:
    - input_frame_bits.npy: original source frame bits
    - encoded_bits.npy: output FEC-encoded bits
    - padding_bits.npy: padding bits array
    - metadata.json: JSON metadata, parameters, block telemetry, and SHA-256 checksums

    Args:
        record: FECRecord instance.
        output_dir: Base directory path.

    Returns:
        Path to the saved directory.
    """
    validate_fec_record(record)

    base_dir = Path(output_dir)
    target_dir = base_dir / record.fec_record_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save NumPy bit arrays
    np.save(target_dir / "input_frame_bits.npy", record.input_bits)
    np.save(target_dir / "encoded_bits.npy", record.encoded_bits)
    np.save(target_dir / "padding_bits.npy", record.padding_bits)

    # Save optional intermediate stages if present
    for stage_name, stage_bits in record.intermediate_stages.items():
        np.save(target_dir / f"{stage_name}.npy", stage_bits)

    # 2. Save metadata JSON
    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)

    return target_dir


def load_fec_record(fec_dir: Path | str) -> FECRecord:
    """Load an FECRecord from its serialized directory.

    Args:
        fec_dir: Path to directory containing FEC files.

    Returns:
        Reconstructed FECRecord.
    """
    pdir = Path(fec_dir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"FEC directory not found: {pdir}")

    meta_path = pdir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json not found in {pdir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    input_bits = np.load(pdir / "input_frame_bits.npy")
    encoded_bits = np.load(pdir / "encoded_bits.npy")
    padding_bits = np.load(pdir / "padding_bits.npy")

    # Load any intermediate stage arrays if present
    intermediates: dict[str, np.ndarray] = {}
    for stage_file in pdir.glob("*.npy"):
        stem = stage_file.stem
        if stem not in ("input_frame_bits", "encoded_bits", "padding_bits"):
            intermediates[stem] = np.load(stage_file)

    record = FECRecord(
        fec_record_id=meta["fec_record_id"],
        frame_id=meta["frame_id"],
        fec_type=meta["fec_type"],
        fec_profile=meta.get("fec_profile"),
        input_bits=input_bits,
        encoded_bits=encoded_bits,
        input_bit_length=meta["input_bit_length"],
        encoded_bit_length=meta["encoded_bit_length"],
        nominal_code_rate=meta["nominal_code_rate"],
        effective_code_rate=meta["effective_code_rate"],
        padding_bits=padding_bits,
        padding_length=meta.get("padding_length", len(padding_bits)),
        block_count=meta.get("block_count", 1),
        block_boundaries=meta.get("block_boundaries", []),
        parameters=meta.get("parameters", {}),
        metadata=meta.get("metadata", {}),
        input_sha256=meta.get("input_sha256", ""),
        encoded_sha256=meta.get("encoded_sha256", ""),
        intermediate_stages=intermediates,
    )

    validate_fec_record(record)
    return record


def save_fec_batch(
    records: Iterable[FECRecord],
    output_dir: Path | str,
) -> list[Path]:
    """Save an iterable of FECRecord instances to disk.

    Args:
        records: Iterable of FECRecord.
        output_dir: Base directory path.

    Returns:
        List of saved directory paths.
    """
    saved = []
    for rec in records:
        path = save_fec_record(rec, output_dir=output_dir)
        saved.append(path)
    return saved
