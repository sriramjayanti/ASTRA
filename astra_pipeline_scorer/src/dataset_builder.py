"""
dataset_builder.py
Dataset builder and group-aware dataset splitting for ASTRA Stage 11.
Enforces strict group splitting by source signal_id to prevent data leakage.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import os
import json
from .models import PipelinePathCandidate
from .feature_builder import PipelineFeatureBuilder
from .feature_schema import PIPELINE_FEATURE_COLUMNS


def split_signals_by_group(
    signal_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split unique signal IDs into train, val, and test sets.
    Ensures that every candidate from a given signal belongs to exactly one split.
    """
    unique_signals = sorted(list(set(signal_ids)))
    np.random.seed(seed)
    shuffled = np.random.permutation(unique_signals).tolist()

    n = len(shuffled)
    n_train = int(np.round(n * train_ratio))
    n_val = int(np.round(n * val_ratio))

    train_signals = set(shuffled[:n_train])
    val_signals = set(shuffled[n_train:n_train + n_val])
    test_signals = set(shuffled[n_train + n_val:])

    # Verify zero leakage
    assert len(train_signals.intersection(val_signals)) == 0, "Leakage between train and val signals!"
    assert len(train_signals.intersection(test_signals)) == 0, "Leakage between train and test signals!"
    assert len(val_signals.intersection(test_signals)) == 0, "Leakage between val and test signals!"

    return list(train_signals), list(val_signals), list(test_signals)


class PipelineDatasetBuilder:
    """
    Constructs multi-candidate tabular datasets for Stage 11 XGBoost training and evaluation.
    """

    def __init__(self):
        self.feature_builder = PipelineFeatureBuilder()
        self.columns = list(PIPELINE_FEATURE_COLUMNS)

    def build_dataset(
        self,
        candidate_list: List[PipelinePathCandidate],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Build feature matrices (X_train, y_train), (X_val, y_val), (X_test, y_test)
        and associated metadata dictionaries with group-aware signal splitting.
        """
        if not candidate_list:
            raise ValueError("Candidate list cannot be empty for dataset generation!")

        signal_ids = [c.signal_id for c in candidate_list]
        train_sigs, val_sigs, test_sigs = split_signals_by_group(
            signal_ids, train_ratio, val_ratio, test_ratio, seed
        )

        train_sigs_set = set(train_sigs)
        val_sigs_set = set(val_sigs)
        test_sigs_set = set(test_sigs)

        train_rows, val_rows, test_rows = [], [], []
        train_y, val_y, test_y = [], [], []
        train_meta, val_meta, test_meta = [], [], []

        for cand in candidate_list:
            f_vec = self.feature_builder.extract_feature_vector(cand)
            label = int(cand.candidate_correct) if cand.candidate_correct is not None else 0
            meta = {
                "signal_id": cand.signal_id,
                "candidate_path_id": cand.candidate_path_id,
                "candidate_id": cand.candidate_id,
                "modulation": cand.modulation,
                "symbol_rate_hz": cand.symbol_rate_hz,
            }

            if cand.signal_id in train_sigs_set:
                train_rows.append(f_vec)
                train_y.append(label)
                train_meta.append(meta)
            elif cand.signal_id in val_sigs_set:
                val_rows.append(f_vec)
                val_y.append(label)
                val_meta.append(meta)
            else:
                test_rows.append(f_vec)
                test_y.append(label)
                test_meta.append(meta)

        return {
            "X_train": np.array(train_rows, dtype=np.float32) if train_rows else np.empty((0, len(self.columns)), dtype=np.float32),
            "y_train": np.array(train_y, dtype=np.int32) if train_y else np.empty((0,), dtype=np.int32),
            "train_meta": train_meta,
            "X_val": np.array(val_rows, dtype=np.float32) if val_rows else np.empty((0, len(self.columns)), dtype=np.float32),
            "y_val": np.array(val_y, dtype=np.int32) if val_y else np.empty((0,), dtype=np.int32),
            "val_meta": val_meta,
            "X_test": np.array(test_rows, dtype=np.float32) if test_rows else np.empty((0, len(self.columns)), dtype=np.float32),
            "y_test": np.array(test_y, dtype=np.int32) if test_y else np.empty((0,), dtype=np.int32),
            "test_meta": test_meta,
            "feature_columns": list(self.columns),
            "train_signal_count": len(train_sigs),
            "val_signal_count": len(val_sigs),
            "test_signal_count": len(test_sigs),
        }
