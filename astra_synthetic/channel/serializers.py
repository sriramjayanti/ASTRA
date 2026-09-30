"""
Serialization utilities for ASTRA Channel Impairment Engine (Engine 6).
Saves and loads ChannelRecord instances in folder mode and compact dataset format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import numpy as np

from .models import ChannelRecord
from .validators import validate_channel_record


def save_channel_record(
    record: ChannelRecord,
    output_dir: Path | str,
    compact: bool = False,
    save_clean_iq: bool = True,
) -> Path:
    """Save a ChannelRecord to disk.

    Modes:
    1. Directory mode (`compact=False`):
       Creates `<output_dir>/<channel_record_id>/` containing:
       - clean_iq.npy: pristine input IQ (optional)
       - impaired_iq.npy: corrupted output IQ
       - metadata.json: full impairment parameters, stage powers, and SHA-256 hashes

    2. Compact mode (`compact=True`):
       Creates:
       - `<output_dir>/impaired_iq/<channel_record_id>.npy`
       - `<output_dir>/metadata/<channel_record_id>.json`

    Args:
        record: ChannelRecord instance.
        output_dir: Base directory path.
        compact: If True, writes in compact dataset mode.
        save_clean_iq: If True, writes clean_iq.npy in folder mode.

    Returns:
        Path to the saved directory or .npy file.
    """
    validate_channel_record(record)
    base_dir = Path(output_dir)

    if compact:
        iq_dir = base_dir / "impaired_iq"
        meta_dir = base_dir / "metadata"
        iq_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        npy_path = iq_dir / f"{record.channel_record_id}.npy"
        np.save(npy_path, record.impaired_iq)

        meta_path = meta_dir / f"{record.channel_record_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        return npy_path

    target_dir = base_dir / record.channel_record_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save numpy arrays
    if save_clean_iq:
        np.save(target_dir / "clean_iq.npy", record.clean_iq)
    np.save(target_dir / "impaired_iq.npy", record.impaired_iq)

    # 2. Save metadata JSON
    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)

    return target_dir


def load_channel_record(chan_path: Path | str) -> ChannelRecord:
    """Load a ChannelRecord from a serialized directory or JSON metadata file."""
    path = Path(chan_path)

    if path.is_file() and path.suffix == ".json":
        meta_path = path
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        rid = meta["channel_record_id"]
        base_dir = meta_path.parent.parent
        compact_npy = base_dir / "impaired_iq" / f"{rid}.npy"

        if compact_npy.is_file():
            impaired_iq = np.load(compact_npy)
            clean_iq = np.empty(0, dtype=np.complex64)
        else:
            sibling_dir = meta_path.parent
            clean_iq = np.load(sibling_dir / "clean_iq.npy") if (sibling_dir / "clean_iq.npy").is_file() else np.empty(0, dtype=np.complex64)
            impaired_iq = np.load(sibling_dir / "impaired_iq.npy")

    elif path.is_dir():
        meta_path = path / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"metadata.json not found in {path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        clean_iq = np.load(path / "clean_iq.npy") if (path / "clean_iq.npy").is_file() else np.empty(0, dtype=np.complex64)
        impaired_iq = np.load(path / "impaired_iq.npy")
    else:
        raise FileNotFoundError(f"Invalid path for channel record: {path}")

    # Reconstruct multipath taps if present
    mp_taps_raw = meta.get("multipath_taps")
    if mp_taps_raw is not None:
        mp_taps = np.array([complex(r, i) for r, i in mp_taps_raw], dtype=np.complex64)
    else:
        mp_taps = None

    mp_delays_raw = meta.get("multipath_delays_samples")
    mp_delays = np.array(mp_delays_raw, dtype=np.int64) if mp_delays_raw is not None else None

    record = ChannelRecord(
        channel_record_id=meta["channel_record_id"],
        modulation_record_id=meta["modulation_record_id"],
        interleaver_record_id=meta["interleaver_record_id"],
        fec_record_id=meta["fec_record_id"],
        frame_id=meta["frame_id"],
        clean_iq=clean_iq,
        impaired_iq=impaired_iq,
        sample_rate=meta["sample_rate_hz"],
        snr_db_target=meta.get("snr_db_target"),
        snr_db_measured=meta.get("snr_db_measured"),
        cfo_hz=meta.get("cfo_hz", 0.0),
        normalized_cfo=meta.get("normalized_cfo", 0.0),
        phase_offset_rad=meta.get("phase_offset_rad", 0.0),
        timing_offset_samples=meta.get("timing_offset_samples", 0.0),
        timing_offset_symbols=meta.get("timing_offset_symbols", 0.0),
        gain_db=meta.get("gain_db", 0.0),
        gain_linear=meta.get("gain_linear", 1.0),
        frequency_drift_hz_per_sec=meta.get("frequency_drift_hz_per_sec", 0.0),
        fading_type=meta.get("fading_type", "none"),
        fading_parameters=meta.get("fading_parameters", {}),
        multipath_enabled=meta.get("multipath_enabled", False),
        multipath_taps=mp_taps,
        multipath_delays_samples=mp_delays,
        interference_enabled=meta.get("interference_enabled", False),
        interference_parameters=meta.get("interference_parameters", {}),
        sample_clock_offset_ppm=meta.get("sample_clock_offset_ppm", 0.0),
        power_stages=meta.get("power_stages", {}),
        impairment_order=meta.get("impairment_order", []),
        parameters=meta.get("parameters", {}),
        metadata=meta.get("metadata", {}),
        clean_iq_sha256=meta.get("clean_iq_sha256", ""),
        impaired_iq_sha256=meta.get("impaired_iq_sha256", ""),
    )

    return record


def save_channel_batch(
    records: Iterable[ChannelRecord],
    output_dir: Path | str,
    compact: bool = False,
) -> list[Path]:
    """Save an iterable of ChannelRecord instances to disk."""
    saved = []
    for rec in records:
        path = save_channel_record(rec, output_dir=output_dir, compact=compact)
        saved.append(path)
    return saved
