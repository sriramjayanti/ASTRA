"""
Serialization utilities for ASTRA Modulation Engine (Engine 5).
Saves and loads ModulationRecord instances in folder mode and compact dataset format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import numpy as np

from .models import ModulationRecord
from .validators import validate_modulation_record


def save_modulation_record(
    record: ModulationRecord,
    output_dir: Path | str,
    compact: bool = False,
) -> Path:
    """Save a ModulationRecord to disk.

    Modes:
    1. Directory mode (`compact=False`):
       Creates `<output_dir>/<modulation_record_id>/` containing:
       - input_bits.npy: source transmitted bits
       - symbol_indices.npy: mapped symbol integer indices
       - ideal_symbols.npy: ideal complex constellation symbols
       - clean_iq.npy: pulse-shaped baseband IQ samples (complex64)
       - metadata.json: full parameters, timing/rate metadata, and SHA-256 hashes

    2. Compact mode (`compact=True`):
       Creates:
       - `<output_dir>/clean_iq/<modulation_record_id>.npy`
       - `<output_dir>/metadata/<modulation_record_id>.json`

    Args:
        record: ModulationRecord instance.
        output_dir: Base directory path.
        compact: If True, writes in compact dataset mode.

    Returns:
        Path to the saved directory or .npy file.
    """
    validate_modulation_record(record)
    base_dir = Path(output_dir)

    if compact:
        iq_dir = base_dir / "clean_iq"
        meta_dir = base_dir / "metadata"
        iq_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        npy_path = iq_dir / f"{record.modulation_record_id}.npy"
        np.save(npy_path, record.clean_iq)

        meta_path = meta_dir / f"{record.modulation_record_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        return npy_path

    target_dir = base_dir / record.modulation_record_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save numpy arrays
    np.save(target_dir / "input_bits.npy", record.input_bits)
    np.save(target_dir / "symbol_indices.npy", record.symbol_indices)
    np.save(target_dir / "ideal_symbols.npy", record.ideal_symbols)
    np.save(target_dir / "clean_iq.npy", record.clean_iq)

    # 2. Save metadata JSON
    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)

    return target_dir


def load_modulation_record(mod_path: Path | str) -> ModulationRecord:
    """Load a ModulationRecord from a serialized directory or JSON metadata file."""
    path = Path(mod_path)

    if path.is_file() and path.suffix == ".json":
        meta_path = path
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        rid = meta["modulation_record_id"]
        base_dir = meta_path.parent.parent
        compact_npy = base_dir / "clean_iq" / f"{rid}.npy"

        if compact_npy.is_file():
            clean_iq = np.load(compact_npy)
            input_bits = np.empty(0, dtype=np.uint8)
            symbol_indices = np.empty(0, dtype=np.int64)
            ideal_symbols = np.empty(0, dtype=np.complex64)
        else:
            sibling_dir = meta_path.parent
            input_bits = np.load(sibling_dir / "input_bits.npy") if (sibling_dir / "input_bits.npy").is_file() else np.empty(0, dtype=np.uint8)
            symbol_indices = np.load(sibling_dir / "symbol_indices.npy") if (sibling_dir / "symbol_indices.npy").is_file() else np.empty(0, dtype=np.int64)
            ideal_symbols = np.load(sibling_dir / "ideal_symbols.npy") if (sibling_dir / "ideal_symbols.npy").is_file() else np.empty(0, dtype=np.complex64)
            clean_iq = np.load(sibling_dir / "clean_iq.npy")

    elif path.is_dir():
        meta_path = path / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"metadata.json not found in {path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        input_bits = np.load(path / "input_bits.npy")
        symbol_indices = np.load(path / "symbol_indices.npy")
        ideal_symbols = np.load(path / "ideal_symbols.npy")
        clean_iq = np.load(path / "clean_iq.npy")
    else:
        raise FileNotFoundError(f"Invalid path for modulation record: {path}")

    pad_len = meta.get("mapping_padding_length", 0)
    pad_bits = np.zeros(pad_len, dtype=np.uint8)

    record = ModulationRecord(
        modulation_record_id=meta["modulation_record_id"],
        interleaver_record_id=meta["interleaver_record_id"],
        fec_record_id=meta["fec_record_id"],
        frame_id=meta["frame_id"],
        modulation_type=meta["modulation_type"],
        modulation_family=meta["modulation_family"],
        modulation_order=meta["modulation_order"],
        bits_per_symbol=meta["bits_per_symbol"],
        input_bits=input_bits,
        mapped_bit_length=meta["mapped_bit_length"],
        mapping_padding_bits=pad_bits,
        mapping_padding_length=pad_len,
        symbol_indices=symbol_indices,
        ideal_symbols=ideal_symbols,
        symbol_count=meta["symbol_count"],
        symbol_rate=meta["symbol_rate_hz"],
        sample_rate=meta["sample_rate_hz"],
        samples_per_symbol=meta["samples_per_symbol"],
        pulse_shape=meta["pulse_shape"],
        rolloff=meta.get("rolloff"),
        filter_span_symbols=meta.get("filter_span_symbols"),
        filter_group_delay_samples=meta.get("filter_group_delay_samples", 0),
        clean_iq=clean_iq,
        clean_iq_sample_count=meta["clean_iq_sample_count"],
        duration_seconds=meta["duration_seconds"],
        average_iq_power=meta["average_iq_power"],
        parameters=meta.get("parameters", {}),
        metadata=meta.get("metadata", {}),
        input_sha256=meta.get("input_sha256", ""),
        iq_sha256=meta.get("iq_sha256", ""),
    )

    return record


def save_modulation_batch(
    records: Iterable[ModulationRecord],
    output_dir: Path | str,
    compact: bool = False,
) -> list[Path]:
    """Save an iterable of ModulationRecord instances to disk."""
    saved = []
    for rec in records:
        path = save_modulation_record(rec, output_dir=output_dir, compact=compact)
        saved.append(path)
    return saved
