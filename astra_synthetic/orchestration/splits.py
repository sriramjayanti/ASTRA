"""
Dataset Splitting and Leakage Prevention Engine for ASTRA (Engine 8).
Enforces source-chain level disjointness across train, validation, and test splits.
"""

from __future__ import annotations

import hashlib
from typing import Sequence
from astra_synthetic.payload.validators import ValidationError
from .models import DatasetRecord


def assign_split_by_chain_id(
    source_chain_id: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    salt: str = "astra_v1_split",
) -> str:
    """Deterministically assign a source_chain_id to 'train', 'validation', or 'test'.

    Uses cryptographic hash of source_chain_id to ensure:
    1. Every record with the same source_chain_id ALWAYS maps to the identical split.
    2. Derivations and windows never leak across splits.
    3. Split ratios are strictly respected over large counts.
    """
    total = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    hash_val = int(hashlib.sha256(f"{salt}:{source_chain_id}".encode("utf-8")).hexdigest()[:8], 16)
    uniform_val = (hash_val % 100000) / 100000.0

    if uniform_val < train_ratio:
        return "train"
    elif uniform_val < (train_ratio + val_ratio):
        return "validation"
    else:
        return "test"


def verify_no_split_leakage(records: Sequence[DatasetRecord]) -> tuple[bool, dict[str, Any]]:
    """Verify that train, validation, and test splits share ZERO common source_chain_ids.

    Args:
        records: List or sequence of DatasetRecord instances.

    Returns:
        tuple (is_valid, leakage_report_dict)

    Raises:
        ValidationError: If any cross-split leakage is detected.
    """
    splits: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
        "other": set(),
    }

    for r in records:
        sp = (r.split or "other").lower()
        if sp in splits:
            splits[sp].add(r.source_chain_id)
        else:
            splits["other"].add(r.source_chain_id)

    train_val_overlap = splits["train"].intersection(splits["validation"])
    train_test_overlap = splits["train"].intersection(splits["test"])
    val_test_overlap = splits["validation"].intersection(splits["test"])

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    report = {
        "leakage_detected": has_leakage,
        "train_unique_chains": len(splits["train"]),
        "validation_unique_chains": len(splits["validation"]),
        "test_unique_chains": len(splits["test"]),
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
        "sample_leaked_chains": list(train_val_overlap.union(train_test_overlap).union(val_test_overlap))[:10],
    }

    if has_leakage:
        raise ValidationError(
            f"CRITICAL SPLIT LEAKAGE DETECTED: {len(train_val_overlap)} train/val, "
            f"{len(train_test_overlap)} train/test, {len(val_test_overlap)} val/test overlaps!"
        )

    return True, report
