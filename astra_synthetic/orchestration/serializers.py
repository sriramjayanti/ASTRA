"""
Dataset Serializers and Package Exporters for ASTRA Orchestration (Engine 8).
Handles dataset persistence, blind test package export, and evaluation truth separation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence
import pandas as pd

from .models import DatasetRecord, DatasetMetadata


def export_blind_test_package(
    records: Sequence[DatasetRecord],
    output_dir: Path | str,
) -> Path:
    """Export a clean blind test package containing ONLY capture files and visible metadata.

    ABSENT of any hidden truth, modulation classes, SNR truth, FEC, or payload labels.
    """
    out_dir = Path(output_dir)
    cap_dir = out_dir / "captures"
    vis_dir = out_dir / "visible_metadata"

    cap_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    blind_rows = []

    for r in records:
        src_file = Path(r.capture_path)
        dest_file = cap_dir / src_file.name
        shutil.copy2(src_file, dest_file)

        src_vis = Path(r.visible_metadata_path)
        dest_vis = vis_dir / src_vis.name
        shutil.copy2(src_vis, dest_vis)

        # Build clean blind manifest (no labels)
        blind_rows.append({
            "dataset_record_id": r.dataset_record_id,
            "source_chain_id": r.source_chain_id,
            "capture_path": str(dest_file.relative_to(out_dir)),
            "visible_metadata_path": str(dest_vis.relative_to(out_dir)),
            "file_format": r.labels.get("file_format", ""),
            "storage_dtype": r.labels.get("storage_dtype", ""),
        })

    blind_manifest = out_dir / "manifest.csv"
    pd.DataFrame(blind_rows).to_csv(blind_manifest, index=False)

    return out_dir


def export_evaluation_truth_package(
    records: Sequence[DatasetRecord],
    output_dir: Path | str,
) -> Path:
    """Export the secret ground-truth package used by the automated evaluator to score blind predictions."""
    out_dir = Path(output_dir)
    truth_dir = out_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = []

    for r in records:
        src_truth = Path(r.truth_metadata_path)
        dest_truth = truth_dir / src_truth.name
        shutil.copy2(src_truth, dest_truth)

        eval_rows.append({
            "dataset_record_id": r.dataset_record_id,
            "source_chain_id": r.source_chain_id,
            "truth_metadata_path": str(dest_truth.relative_to(out_dir)),
            "modulation_truth": r.labels.get("modulation_type", ""),
            "symbol_rate_truth": r.labels.get("symbol_rate", 0.0),
            "sample_rate_truth": r.labels.get("sample_rate", 0.0),
            "snr_db_truth": r.labels.get("snr_db", None),
            "cfo_hz_truth": r.labels.get("cfo_hz", 0.0),
            "fec_family_truth": r.labels.get("fec_family", ""),
            "interleaver_family_truth": r.labels.get("interleaver_family", ""),
            "payload_length_truth": r.labels.get("payload_length", 0),
        })

    eval_manifest = out_dir / "evaluation_manifest.csv"
    pd.DataFrame(eval_rows).to_csv(eval_manifest, index=False)

    return out_dir
