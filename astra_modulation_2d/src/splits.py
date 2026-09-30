"""
Source-Level Dataset Splitting and Zero-Leakage Verification for ASTRA.

Guarantees:
1. Signal files are assigned to train / val / test BEFORE window slicing.
2. Train, validation, and test sets are strictly disjoint (Zero Leakage).
3. Split files are serializable and consumable by both 1D and 2D models for fair fusion comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("ASTRA_SPLITS")


def verify_disjoint_splits(
    train_ids: Sequence[Union[int, str]],
    val_ids: Sequence[Union[int, str]],
    test_ids: Sequence[Union[int, str]],
) -> None:
    """Verifies that train, validation, and test source signal IDs are 100% disjoint."""
    s_train = set(train_ids)
    s_val = set(val_ids)
    s_test = set(test_ids)

    train_val_overlap = s_train.intersection(s_val)
    train_test_overlap = s_train.intersection(s_test)
    val_test_overlap = s_val.intersection(s_test)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        err_msg = (
            f"CRITICAL SPLIT LEAKAGE DETECTED!\n"
            f"Train/Val Overlap: {len(train_val_overlap)}\n"
            f"Train/Test Overlap: {len(train_test_overlap)}\n"
            f"Val/Test Overlap: {len(val_test_overlap)}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info("Split verification PASSED: Zero source signal leakage across all partitions.")


def save_split_json(
    split_path: Union[str, Path],
    train_ids: List[Union[int, str]],
    val_ids: List[Union[int, str]],
    test_ids: List[Union[int, str]],
    metadata: Optional[Dict] = None,
) -> str:
    """Saves split partitions to a JSON manifest with checksum."""
    verify_disjoint_splits(train_ids, val_ids, test_ids)
    split_p = Path(split_path)
    split_p.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "train": [str(x) for x in train_ids],
        "validation": [str(x) for x in val_ids],
        "test": [str(x) for x in test_ids],
        "metadata": metadata or {},
    }
    raw_str = json.dumps(content, indent=2)
    split_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    content["split_hash"] = split_hash

    with open(split_p, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)

    logger.info(f"Saved split JSON to {split_p} (hash: {split_hash[:12]})")
    return split_hash


def load_split_json(split_path: Union[str, Path]) -> Tuple[List[str], List[str], List[str]]:
    """Loads split partitions from a JSON manifest."""
    split_p = Path(split_path)
    if not split_p.exists():
        raise FileNotFoundError(f"Split file not found: {split_p}")

    with open(split_p, "r", encoding="utf-8") as f:
        content = json.load(f)

    train_ids = [str(x) for x in content.get("train", [])]
    val_ids = [str(x) for x in content.get("validation", [])]
    test_ids = [str(x) for x in content.get("test", [])]

    verify_disjoint_splits(train_ids, val_ids, test_ids)
    return train_ids, val_ids, test_ids
