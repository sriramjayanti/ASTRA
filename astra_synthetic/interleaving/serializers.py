"""
Serialization utilities for ASTRA Interleaving Engine.
Saves and loads InterleaverRecord instances in both folder and compact dataset formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import numpy as np

from .models import InterleaverRecord
from .validators import validate_interleaver_record


def save_interleaver_record(
    record: InterleaverRecord,
    output_dir: Path | str,
    save_permutation: bool = True,
    compact: bool = False,
) -> Path:
    """Save an InterleaverRecord to disk.

    Modes:
    1. Directory mode (`compact=False`):
       Creates `<output_dir>/<interleaver_record_id>/` containing:
       - input_bits.npy: original FEC-encoded bitstream
       - interleaved_bits.npy: output interleaved bitstream
       - padding_bits.npy: padding bits array
       - permutation.npy: (optional) index permutation array
       - metadata.json: full parameters, block telemetry, and SHA-256 hashes

    2. Compact mode (`compact=True`):
       Creates:
       - `<output_dir>/interleaved_bits/<interleaver_record_id>.npy`
       - `<output_dir>/metadata/<interleaver_record_id>.json`

    Args:
        record: InterleaverRecord instance.
        output_dir: Base directory path.
        save_permutation: Whether to save the permutation array if present.
        compact: If True, writes in compact dataset mode.

    Returns:
        Path to the saved directory or .npy file.
    """
    validate_interleaver_record(record)
    base_dir = Path(output_dir)

    if compact:
        bits_dir = base_dir / "interleaved_bits"
        meta_dir = base_dir / "metadata"
        bits_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        npy_path = bits_dir / f"{record.interleaver_record_id}.npy"
        np.save(npy_path, record.interleaved_bits)

        meta_path = meta_dir / f"{record.interleaver_record_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        return npy_path

    target_dir = base_dir / record.interleaver_record_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save bit arrays
    np.save(target_dir / "input_bits.npy", record.input_bits)
    np.save(target_dir / "interleaved_bits.npy", record.interleaved_bits)
    np.save(target_dir / "padding_bits.npy", record.padding_bits)

    if save_permutation and record.permutation is not None:
        np.save(target_dir / "permutation.npy", record.permutation)

    # 2. Save metadata JSON
    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)

    return target_dir


def load_interleaver_record(int_path: Path | str) -> InterleaverRecord:
    """Load an InterleaverRecord from a serialized folder or metadata JSON file.

    Args:
        int_path: Path to directory containing interleaver files or direct path to metadata JSON.

    Returns:
        Reconstructed InterleaverRecord.
    """
    path = Path(int_path)

    if path.is_file() and path.suffix == ".json":
        meta_path = path
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        rid = meta["interleaver_record_id"]
        # Look for corresponding .npy file in compact mode or sibling directory
        base_dir = meta_path.parent.parent
        compact_npy = base_dir / "interleaved_bits" / f"{rid}.npy"
        if compact_npy.is_file():
            interleaved_bits = np.load(compact_npy)
            input_bits = np.empty(0, dtype=np.uint8)
            padding_bits = np.empty(0, dtype=np.uint8)
            permutation = None
        else:
            # Sibling directory check
            sibling_dir = meta_path.parent
            input_bits = np.load(sibling_dir / "input_bits.npy") if (sibling_dir / "input_bits.npy").is_file() else np.empty(0, dtype=np.uint8)
            interleaved_bits = np.load(sibling_dir / "interleaved_bits.npy")
            padding_bits = np.load(sibling_dir / "padding_bits.npy") if (sibling_dir / "padding_bits.npy").is_file() else np.empty(0, dtype=np.uint8)
            perm_p = sibling_dir / "permutation.npy"
            permutation = np.load(perm_p) if perm_p.is_file() else None

    elif path.is_dir():
        meta_path = path / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"metadata.json not found in {path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        input_bits = np.load(path / "input_bits.npy")
        interleaved_bits = np.load(path / "interleaved_bits.npy")
        padding_bits = np.load(path / "padding_bits.npy")

        perm_path = path / "permutation.npy"
        permutation = np.load(perm_path) if perm_path.is_file() else None
    else:
        raise FileNotFoundError(f"Invalid path for interleaver record: {path}")

    record = InterleaverRecord(
        interleaver_record_id=meta["interleaver_record_id"],
        fec_record_id=meta["fec_record_id"],
        frame_id=meta["frame_id"],
        interleaver_type=meta["interleaver_type"],
        profile_name=meta.get("profile_name"),
        input_bits=input_bits,
        interleaved_bits=interleaved_bits,
        input_bit_length=meta["input_bit_length"],
        output_bit_length=meta["output_bit_length"],
        padding_bits=padding_bits,
        padding_length=meta.get("padding_length", len(padding_bits)),
        parameters=meta.get("parameters", {}),
        permutation=permutation,
        block_boundaries=meta.get("block_boundaries", []),
        input_sha256=meta.get("input_sha256", ""),
        output_sha256=meta.get("output_sha256", ""),
        metadata=meta.get("metadata", {}),
    )

    return record


def save_interleaver_batch(
    records: Iterable[InterleaverRecord],
    output_dir: Path | str,
    save_permutation: bool = True,
    compact: bool = False,
) -> list[Path]:
    """Save an iterable of InterleaverRecord instances to disk.

    Args:
        records: Iterable of InterleaverRecord.
        output_dir: Base output directory.
        save_permutation: Whether to write permutation.npy.
        compact: If True, writes in compact dataset mode.

    Returns:
        List of saved directory or file paths.
    """
    saved = []
    for rec in records:
        path = save_interleaver_record(
            rec,
            output_dir=output_dir,
            save_permutation=save_permutation,
            compact=compact,
        )
        saved.append(path)
    return saved
