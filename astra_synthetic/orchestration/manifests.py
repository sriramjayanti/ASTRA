"""
Manifest Generator for ASTRA Dataset Orchestration (Engine 8).
Generates master CSV manifests, split-specific manifests, and task-specific training tables
(Modulation, Symbol Rate, FEC, Interleaver, and Bitstream Structure).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence
import pandas as pd

from .models import DatasetRecord


def build_master_manifest(
    records: Sequence[DatasetRecord],
    output_path: Path | str,
) -> Path:
    """Build the comprehensive master manifest (all.csv)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in records:
        lbls = r.labels
        rows.append({
            "dataset_record_id": r.dataset_record_id,
            "source_chain_id": r.source_chain_id,
            "split": r.split,
            "capture_path": r.capture_path,
            "visible_metadata_path": r.visible_metadata_path,
            "truth_metadata_path": r.truth_metadata_path,
            "modulation": lbls.get("modulation_type", ""),
            "symbol_rate": lbls.get("symbol_rate", 0.0),
            "sample_rate": lbls.get("sample_rate", 0.0),
            "snr_db": lbls.get("snr_db", None),
            "cfo_hz": lbls.get("cfo_hz", 0.0),
            "fec_family": lbls.get("fec_family", ""),
            "fec_profile": lbls.get("fec_profile", ""),
            "interleaver_family": lbls.get("interleaver_family", ""),
            "interleaver_profile": lbls.get("interleaver_profile", ""),
            "difficulty": lbls.get("difficulty", "medium"),
            "file_format": lbls.get("file_format", "raw_iq"),
            "storage_dtype": lbls.get("storage_dtype", "float32"),
            "valid": r.valid,
            "record_sha256": r.record_sha256,
        })

    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return p


def build_split_manifests(
    records: Sequence[DatasetRecord],
    manifests_dir: Path | str,
) -> dict[str, Path]:
    """Generate separate train.csv, validation.csv, and test.csv manifests."""
    m_dir = Path(manifests_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    splits_dict: dict[str, list[DatasetRecord]] = {"train": [], "validation": [], "test": []}
    for r in records:
        sp = (r.split or "train").lower()
        if sp in splits_dict:
            splits_dict[sp].append(r)

    saved_paths: dict[str, Path] = {}
    for sp_name, recs in splits_dict.items():
        sp_path = m_dir / f"{sp_name}.csv"
        build_master_manifest(recs, sp_path)
        saved_paths[sp_name] = sp_path

    return saved_paths


def build_task_specific_manifests(
    records: Sequence[DatasetRecord],
    manifests_dir: Path | str,
) -> dict[str, Path]:
    """Build dedicated task manifests for specialized deep learning models.

    Exports:
    - modulation.csv: for 1D ResNet / Transformer modulation classification.
    - fec.csv: for blind FEC code family/profile estimation.
    - interleaver.csv: for blind interleaver parameter detection.
    - bitstream.csv: for frame boundary and sync detection models.
    """
    m_dir = Path(manifests_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    mod_rows, fec_rows, int_rows, bit_rows = [], [], [], []

    for r in records:
        lbls = r.labels

        # 1. Modulation Manifest
        mod_rows.append({
            "dataset_record_id": r.dataset_record_id,
            "source_chain_id": r.source_chain_id,
            "capture_path": r.capture_path,
            "modulation": lbls.get("modulation_type", ""),
            "symbol_rate": lbls.get("symbol_rate", 0.0),
            "sample_rate": lbls.get("sample_rate", 0.0),
            "snr_db": lbls.get("snr_db", None),
            "cfo_hz": lbls.get("cfo_hz", 0.0),
            "split": r.split,
        })

        # 2. FEC Manifest
        fec_rows.append({
            "dataset_record_id": r.dataset_record_id,
            "source_chain_id": r.source_chain_id,
            "fec_family": lbls.get("fec_family", ""),
            "fec_profile": lbls.get("fec_profile", ""),
            "code_rate": lbls.get("code_rate", 1.0),
            "split": r.split,
        })

        # 3. Interleaver Manifest
        int_rows.append({
            "dataset_record_id": r.dataset_record_id,
            "source_chain_id": r.source_chain_id,
            "interleaver_family": lbls.get("interleaver_family", ""),
            "interleaver_profile": lbls.get("interleaver_profile", ""),
            "split": r.split,
        })

        # 4. Bitstream Structure Manifest
        bit_rows.append({
            "dataset_record_id": r.dataset_record_id,
            "source_chain_id": r.source_chain_id,
            "sync_start": lbls.get("sync_start", 0),
            "sync_length": lbls.get("sync_length", 32),
            "header_start": lbls.get("header_start", 32),
            "header_length": lbls.get("header_length", 56),
            "payload_start": lbls.get("payload_start", 88),
            "payload_length": lbls.get("payload_length", 0),
            "crc_start": lbls.get("crc_start", 0),
            "crc_length": lbls.get("crc_length", 16),
            "split": r.split,
        })

    paths = {}

    p_mod = m_dir / "modulation.csv"
    pd.DataFrame(mod_rows).to_csv(p_mod, index=False)
    paths["modulation"] = p_mod

    p_fec = m_dir / "fec.csv"
    pd.DataFrame(fec_rows).to_csv(p_fec, index=False)
    paths["fec"] = p_fec

    p_int = m_dir / "interleaver.csv"
    pd.DataFrame(int_rows).to_csv(p_int, index=False)
    paths["interleaver"] = p_int

    p_bit = m_dir / "bitstream.csv"
    pd.DataFrame(bit_rows).to_csv(p_bit, index=False)
    paths["bitstream"] = p_bit

    return paths
