"""
ASTRA Synthetic Engine 8: Ground-Truth / Dataset Orchestrator.
Connects Engines 1 through 7 into a scalable, reproducible, and verifiable dataset generation pipeline.
"""

from .models import DatasetRecord, DatasetMetadata
from .generator import ASTRASyntheticDatasetGenerator
from .seed_manager import SeedManager, ChainSeeds
from .chain_ids import (
    make_chain_id,
    make_dataset_record_id,
    make_payload_id,
    make_frame_id,
    make_fec_id,
    make_interleaver_id,
    make_modulation_id,
    make_channel_id,
    make_capture_id,
)
from .balancing import ProfileSampler
from .splits import assign_split_by_chain_id, verify_no_split_leakage
from .truth_builder import build_full_truth_document
from .visible_metadata import build_visible_metadata_document
from .validation import validate_full_synthetic_chain
from .manifests import build_master_manifest, build_split_manifests, build_task_specific_manifests
from .statistics import compute_dataset_statistics, generate_dataset_reports
from .serializers import export_blind_test_package, export_evaluation_truth_package

__all__ = [
    "DatasetRecord",
    "DatasetMetadata",
    "ASTRASyntheticDatasetGenerator",
    "SeedManager",
    "ChainSeeds",
    "make_chain_id",
    "make_dataset_record_id",
    "make_payload_id",
    "make_frame_id",
    "make_fec_id",
    "make_interleaver_id",
    "make_modulation_id",
    "make_channel_id",
    "make_capture_id",
    "ProfileSampler",
    "assign_split_by_chain_id",
    "verify_no_split_leakage",
    "build_full_truth_document",
    "build_visible_metadata_document",
    "validate_full_synthetic_chain",
    "build_master_manifest",
    "build_split_manifests",
    "build_task_specific_manifests",
    "compute_dataset_statistics",
    "generate_dataset_reports",
    "export_blind_test_package",
    "export_evaluation_truth_package",
]
