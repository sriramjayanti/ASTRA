"""
Serialization and file persistence utilities for ASTRA Framing Engine.
Supports multi-format individual frame folders, batch frames, and continuous multi-frame streams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import numpy as np

from ..payload.models import bits_to_bytes
from .models import FrameRecord, FrameStreamRecord
from .validators import validate_frame_record


def save_frame(
    record: FrameRecord,
    output_dir: Path | str,
    save_txt: bool = True,
    max_txt_bits: int = 8192,
) -> Path:
    """Save an individual FrameRecord and its sub-arrays to disk.

    Creates `<output_dir>/<frame_id>/` containing:
    - frame.bin: packed raw frame bytes
    - frame_bits.npy: full frame bit array
    - frame_bits.txt: text bitstring (optional)
    - sync_bits.npy: sync region bits
    - header_bits.npy: header region bits
    - payload_bits.npy: payload region bits
    - crc_bits.npy: CRC region bits
    - metadata.json: comprehensive ground truth metadata

    Args:
        record: FrameRecord to serialize.
        output_dir: Base directory path.
        save_txt: Whether to write human-readable .txt bitstring.
        max_txt_bits: Maximum bits threshold for writing .txt file.

    Returns:
        Path to the saved frame directory.
    """
    validate_frame_record(record)

    base_dir = Path(output_dir)
    target_dir = base_dir / record.frame_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save binary representations
    with open(target_dir / "frame.bin", "wb") as f:
        f.write(record.frame_bytes)

    # 2. Save NumPy arrays
    np.save(target_dir / "frame_bits.npy", record.frame_bits)
    np.save(target_dir / "sync_bits.npy", record.sync_bits)
    np.save(target_dir / "header_bits.npy", record.header_bits)
    np.save(target_dir / "payload_bits.npy", record.payload_bits)
    np.save(target_dir / "crc_bits.npy", record.crc_bits)

    # 3. Save text bitstring if within size limit
    if save_txt and record.frame_bit_length <= max_txt_bits:
        bit_str = "".join(str(b) for b in record.frame_bits)
        with open(target_dir / "frame_bits.txt", "w", encoding="utf-8") as f:
            f.write(bit_str)

    # 4. Save metadata JSON
    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)

    return target_dir


def load_frame(frame_dir: Path | str) -> FrameRecord:
    """Load and reconstruct a FrameRecord from a serialized directory.

    Args:
        frame_dir: Path to directory containing frame files.

    Returns:
        Reconstructed FrameRecord.
    """
    pdir = Path(frame_dir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {pdir}")

    meta_path = pdir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json not found in {pdir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load arrays
    frame_bits = np.load(pdir / "frame_bits.npy")
    sync_bits = np.load(pdir / "sync_bits.npy")
    header_bits = np.load(pdir / "header_bits.npy")
    payload_bits = np.load(pdir / "payload_bits.npy")
    crc_bits = np.load(pdir / "crc_bits.npy")

    with open(pdir / "frame.bin", "rb") as f:
        frame_bytes = f.read()

    record = FrameRecord(
        frame_id=meta["frame_id"],
        payload_id=meta["payload_id"],
        frame_bits=frame_bits,
        frame_bytes=frame_bytes,
        frame_bit_length=meta["frame_bit_length"],
        sync_bits=sync_bits,
        header_bits=header_bits,
        payload_bits=payload_bits,
        crc_bits=crc_bits,
        sync_start=meta["sync"]["start"],
        sync_length=meta["sync"]["length"],
        header_start=meta["header"]["start"],
        header_length=meta["header"]["length"],
        payload_start=meta["payload"]["start"],
        payload_length=meta["payload"]["length"],
        crc_start=meta["crc"]["start"],
        crc_length=meta["crc"]["length"],
        sync_type=meta["sync"]["type"],
        sync_value=meta["sync"]["value"],
        header_schema=meta["header"]["schema"],
        header_fields=meta["header"]["fields"],
        header_field_offsets=meta["header"].get("field_offsets", {}),
        crc_type=meta["crc"]["type"],
        crc_value=meta["crc"]["value"],
        crc_scope=meta["crc"].get("scope", ["header", "payload"]),
        frame_sequence_number=meta.get("frame_sequence_number", 0),
        metadata=meta.get("metadata", {}),
        sha256=meta.get("sha256", ""),
    )

    validate_frame_record(record)
    return record


def save_frame_batch(
    records: Iterable[FrameRecord],
    output_dir: Path | str,
    save_txt: bool = True,
    max_txt_bits: int = 8192,
) -> list[Path]:
    """Save an iterable of FrameRecord objects to disk.

    Args:
        records: Iterable of FrameRecord instances.
        output_dir: Output base directory.
        save_txt: Whether to save .txt representations.
        max_txt_bits: Bit length limit for .txt.

    Returns:
        List of created directory paths.
    """
    saved = []
    for rec in records:
        path = save_frame(rec, output_dir=output_dir, save_txt=save_txt, max_txt_bits=max_txt_bits)
        saved.append(path)
    return saved


def save_frame_stream(
    stream: FrameStreamRecord,
    output_dir: Path | str,
    base_name: str = "frame_stream",
) -> tuple[Path, Path]:
    """Save a continuous concatenated multi-frame stream and its telemetry.

    Creates:
    - `<output_dir>/<base_name>.npy`: raw NumPy uint8 array of stream bits
    - `<output_dir>/<base_name>.bin`: raw packed bytes
    - `<output_dir>/<base_name>_metadata.json`: telemetry containing all frame start positions and boundaries

    Args:
        stream: FrameStreamRecord instance.
        output_dir: Output directory.
        base_name: Base filename prefix.

    Returns:
        tuple of (npy_path, json_metadata_path)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{base_name}.npy"
    bin_path = out_dir / f"{base_name}.bin"
    json_path = out_dir / f"{base_name}_metadata.json"

    np.save(npy_path, stream.stream_bits)
    with open(bin_path, "wb") as f:
        f.write(stream.stream_bytes)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stream.to_dict(), f, indent=2)

    return npy_path, json_path


def load_frame_stream(
    npy_path: Path | str,
    meta_path: Path | str,
) -> FrameStreamRecord:
    """Load a FrameStreamRecord from .npy and metadata.json files.

    Args:
        npy_path: Path to .npy stream bit file.
        meta_path: Path to metadata.json file.

    Returns:
        Reconstructed FrameStreamRecord.
    """
    npath = Path(npy_path)
    mpath = Path(meta_path)

    if not npath.is_file():
        raise FileNotFoundError(f"Stream bit file not found: {npath}")
    if not mpath.is_file():
        raise FileNotFoundError(f"Stream metadata file not found: {mpath}")

    stream_bits = np.load(npath)
    with open(mpath, "r", encoding="utf-8") as f:
        meta = json.load(f)

    stream_bytes = bits_to_bytes(stream_bits, padding=True)

    return FrameStreamRecord(
        stream_bits=stream_bits,
        stream_bytes=stream_bytes,
        total_bit_length=meta["total_bit_length"],
        frame_ids=meta["frame_ids"],
        frame_start_positions=meta["frame_start_positions"],
        frame_lengths=meta["frame_lengths"],
        sync_positions=meta["sync_positions"],
        header_positions=meta["header_positions"],
        payload_positions=meta["payload_positions"],
        crc_positions=meta["crc_positions"],
        gap_positions=meta.get("gap_positions", []),
        metadata=meta.get("metadata", {}),
        sha256=meta.get("sha256", ""),
    )
