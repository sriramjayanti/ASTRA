"""
Serialization and deserialization utilities for ASTRA Payload Engine.
Saves payloads to disk in standardized multi-format directory structures (.bin, .npy, .json, .txt).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import numpy as np

from .models import PayloadRecord
from .validators import validate_payload_record


def save_payload(
    record: PayloadRecord,
    output_dir: Path | str,
    save_txt: bool = True,
    max_txt_bits: int = 8192,
) -> Path:
    """Save a single PayloadRecord to disk.

    Creates a subfolder named after `record.payload_id` containing:
    - payload.bin: raw binary payload bytes
    - payload_bits.npy: NumPy array of uint8 bits
    - metadata.json: JSON file containing all record attributes and audit metadata
    - payload.txt: (optional) human-readable bit string for inspectability

    Args:
        record: The PayloadRecord to serialize.
        output_dir: Base output directory.
        save_txt: If True, writes human-readable .txt bitstring.
        max_txt_bits: Maximum bit length for which .txt is generated (prevents huge text files).

    Returns:
        Path to the created payload directory.
    """
    validate_payload_record(record)
    
    base_dir = Path(output_dir)
    target_dir = base_dir / record.payload_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw bytes
    bin_path = target_dir / "payload.bin"
    with open(bin_path, "wb") as f:
        f.write(record.payload_bytes)

    # 2. Save NumPy bit array
    npy_path = target_dir / "payload_bits.npy"
    np.save(npy_path, record.payload_bits)

    # 3. Save text bit string if within limit
    if save_txt and record.bit_length <= max_txt_bits:
        txt_path = target_dir / "payload.txt"
        bit_str = "".join(str(b) for b in record.payload_bits)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(bit_str)

    # 4. Save metadata JSON
    meta_path = target_dir / "metadata.json"
    meta_dict = record.to_dict()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return target_dir


def load_payload(payload_dir: Path | str) -> PayloadRecord:
    """Load a PayloadRecord from its serialized directory.

    Args:
        payload_dir: Path to the folder containing payload files.

    Returns:
        Reconstructed PayloadRecord.
    """
    pdir = Path(payload_dir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"Payload directory not found: {pdir}")

    meta_path = pdir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json not found in {pdir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    npy_path = pdir / "payload_bits.npy"
    if npy_path.is_file():
        payload_bits = np.load(npy_path)
    else:
        # Fallback to reconstructing from .bin
        bin_path = pdir / "payload.bin"
        if not bin_path.is_file():
            raise FileNotFoundError(f"Neither payload_bits.npy nor payload.bin found in {pdir}")
        with open(bin_path, "rb") as f:
            raw_bytes = f.read()
        from .models import bytes_to_bits
        payload_bits = bytes_to_bits(raw_bytes)[:meta_dict["bit_length"]]

    bin_path = pdir / "payload.bin"
    if bin_path.is_file():
        with open(bin_path, "rb") as f:
            payload_bytes = f.read()
    else:
        from .models import bits_to_bytes
        payload_bytes = bits_to_bytes(payload_bits, padding=True)

    record = PayloadRecord(
        payload_id=meta_dict["payload_id"],
        payload_type=meta_dict["payload_type"],
        payload_bits=payload_bits,
        payload_bytes=payload_bytes,
        bit_length=meta_dict["bit_length"],
        byte_length=meta_dict["byte_length"],
        seed=meta_dict.get("seed"),
        source_text=meta_dict.get("source_text"),
        pattern=meta_dict.get("pattern"),
        entropy_estimate=meta_dict.get("entropy_estimate", 0.0),
        sha256=meta_dict.get("sha256", ""),
        metadata=meta_dict.get("metadata", {}),
    )

    validate_payload_record(record)
    return record


def save_batch(
    records: Iterable[PayloadRecord],
    output_dir: Path | str,
    save_txt: bool = True,
    max_txt_bits: int = 8192,
) -> list[Path]:
    """Save an iterable of PayloadRecord instances to disk.

    Args:
        records: Iterable of PayloadRecord.
        output_dir: Output base directory.
        save_txt: Whether to write .txt bitstrings.
        max_txt_bits: Max bits for .txt generation.

    Returns:
        List of paths to generated payload directories.
    """
    saved_paths = []
    for rec in records:
        path = save_payload(
            rec,
            output_dir=output_dir,
            save_txt=save_txt,
            max_txt_bits=max_txt_bits,
        )
        saved_paths.append(path)
    return saved_paths
