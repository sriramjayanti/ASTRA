"""
Data models for ASTRA Ground-Truth / Dataset Orchestrator (Engine 8).
Encapsulates end-to-end dataset records, complete source-chain linkages,
dataset run metadata, and split assignments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetRecord:
    """Represents a single end-to-end synthetic dataset example with complete multi-stage ground truth.

    Attributes:
        dataset_record_id: Unique record ID within dataset (e.g. 'ds_rec_00000001').
        source_chain_id: Unified lineage ID linking Engines 1-7 (e.g. 'chain_00000001').
        payload_id: Source PayloadRecord ID.
        frame_id: Source FrameRecord ID.
        fec_record_id: Source FECRecord ID.
        interleaver_record_id: Source InterleaverRecord ID.
        modulation_record_id: Source ModulationRecord ID.
        channel_record_id: Source ChannelRecord ID.
        capture_record_id: Source CaptureRecord ID.
        capture_path: Relative or absolute path to the physical SDR capture file.
        visible_metadata_path: Path to visible capture metadata sidecar.
        truth_metadata_path: Path to hidden synthetic truth metadata sidecar.
        split: Dataset split ('train', 'validation', 'test', 'holdout', 'ood', or None).
        labels: Compact task labels dictionary (modulation, FEC, interleaver, SNR, etc.).
        generation_parameters: Complete dictionary of resolved configuration parameters across all engines.
        source_seed: Base master seed used for this chain.
        generator_versions: Version dictionary of all 7 engines + orchestrator.
        valid: Whether the generated chain passed all validation checks.
        validation_results: Detailed validation results and reference recovery status.
        record_sha256: SHA-256 hash of this record's serialized parameters.
    """
    dataset_record_id: str
    source_chain_id: str
    payload_id: str
    frame_id: str
    fec_record_id: str
    interleaver_record_id: str
    modulation_record_id: str
    channel_record_id: str
    capture_record_id: str
    capture_path: str
    visible_metadata_path: str
    truth_metadata_path: str
    split: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    source_seed: int = 0
    generator_versions: dict[str, str] = field(default_factory=dict)
    valid: bool = True
    validation_results: dict[str, Any] = field(default_factory=dict)
    record_sha256: str = ""

    def __post_init__(self):
        if not self.record_sha256:
            content_str = (
                f"{self.dataset_record_id}:{self.source_chain_id}:{self.capture_path}:"
                f"{json.dumps(self.labels, sort_keys=True)}"
            )
            self.record_sha256 = hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert DatasetRecord to JSON-serializable dictionary."""
        return {
            "dataset_record_id": self.dataset_record_id,
            "source_chain_id": self.source_chain_id,
            "payload_id": self.payload_id,
            "frame_id": self.frame_id,
            "fec_record_id": self.fec_record_id,
            "interleaver_record_id": self.interleaver_record_id,
            "modulation_record_id": self.modulation_record_id,
            "channel_record_id": self.channel_record_id,
            "capture_record_id": self.capture_record_id,
            "capture_path": self.capture_path,
            "visible_metadata_path": self.visible_metadata_path,
            "truth_metadata_path": self.truth_metadata_path,
            "split": self.split,
            "labels": self.labels,
            "generation_parameters": self.generation_parameters,
            "source_seed": self.source_seed,
            "generator_versions": self.generator_versions,
            "valid": self.valid,
            "validation_results": self.validation_results,
            "record_sha256": self.record_sha256,
        }


@dataclass
class DatasetMetadata:
    """Global summary metadata for an entire synthetic dataset release."""
    dataset_name: str
    version: str
    creation_time_utc: str
    master_seed: int
    total_requested: int
    total_valid: int
    total_failed: int
    split_ratios: dict[str, float]
    split_counts: dict[str, int]
    class_distributions: dict[str, dict[str, int]]
    generator_versions: dict[str, str]
    configuration_sha256: str
    storage_root: str
    manifest_paths: dict[str, str] = field(default_factory=dict)
    summary_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "creation_time_utc": self.creation_time_utc,
            "master_seed": self.master_seed,
            "total_requested": self.total_requested,
            "total_valid": self.total_valid,
            "total_failed": self.total_failed,
            "split_ratios": self.split_ratios,
            "split_counts": self.split_counts,
            "class_distributions": self.class_distributions,
            "generator_versions": self.generator_versions,
            "configuration_sha256": self.configuration_sha256,
            "storage_root": self.storage_root,
            "manifest_paths": self.manifest_paths,
            "summary_metrics": self.summary_metrics,
        }
