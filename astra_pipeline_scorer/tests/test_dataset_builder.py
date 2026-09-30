"""
test_dataset_builder.py
Unit tests for candidate dataset construction and ground-truth label generation.
"""

import pytest
import numpy as np
from astra_pipeline_scorer.src.utils import generate_synthetic_candidate_tree
from astra_pipeline_scorer.src.dataset_builder import PipelineDatasetBuilder
from astra_pipeline_scorer.src.label_builder import evaluate_candidate_correctness


def test_training_dataset_generation():
    all_cands = []
    for s_idx in range(6):
        cands, gt = generate_synthetic_candidate_tree(f"sig_{s_idx}", num_competing_candidates=8, seed=s_idx)
        all_cands.extend(cands)

    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)

    assert data["X_train"].shape[1] == len(builder.columns)
    assert len(data["X_train"]) == len(data["y_train"])
    assert len(data["X_val"]) == len(data["y_val"])
    assert len(data["X_test"]) == len(data["y_test"])


def test_candidate_labels_correctness():
    gt = {"modulation": "QPSK", "symbol_rate_hz": 9600.0, "interleaver_family": "block", "fec_family": "convolutional"}
    cand_right = {"modulation": "QPSK", "symbol_rate_hz": 9600.0, "interleaver_family": "block", "fec_family": "convolutional"}
    cand_wrong_mod = {"modulation": "8PSK", "symbol_rate_hz": 9600.0, "interleaver_family": "block", "fec_family": "convolutional"}
    cand_wrong_rate = {"modulation": "QPSK", "symbol_rate_hz": 4800.0, "interleaver_family": "block", "fec_family": "convolutional"}

    assert evaluate_candidate_correctness(cand_right, gt) == 1
    assert evaluate_candidate_correctness(cand_wrong_mod, gt) == 0
    assert evaluate_candidate_correctness(cand_wrong_rate, gt) == 0
