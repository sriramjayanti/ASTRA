"""
test_training.py
Unit tests for XGBoost model training, early stopping, and checkpoint saving/loading.
"""

import pytest
import numpy as np
import tempfile
import os
from astra_pipeline_scorer.src.utils import generate_synthetic_candidate_tree
from astra_pipeline_scorer.src.dataset_builder import PipelineDatasetBuilder
from astra_pipeline_scorer.src.training import train_pipeline_scorer
from astra_pipeline_scorer.src.checkpoint import save_pipeline_model_package, load_pipeline_model_package


def test_xgboost_training_runs():
    all_cands = []
    for s_idx in range(12):
        cands, _ = generate_synthetic_candidate_tree(f"sig_{s_idx}", num_competing_candidates=8, seed=s_idx)
        all_cands.extend(cands)

    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, train_ratio=0.70, val_ratio=0.30, test_ratio=0.0)

    val_sigs = [m["signal_id"] for m in data["val_meta"]]
    model, calib, report, meta = train_pipeline_scorer(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        val_signal_ids=val_sigs,
    )

    assert model is not None
    assert meta["train_samples"] > 0
    assert report.total_signals > 0


def test_checkpoint_save_and_load():
    all_cands = []
    for s_idx in range(8):
        cands, _ = generate_synthetic_candidate_tree(f"sig_{s_idx}", num_competing_candidates=6, seed=s_idx)
        all_cands.extend(cands)

    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, train_ratio=0.70, val_ratio=0.30, test_ratio=0.0)

    model, calib, _, _ = train_pipeline_scorer(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = save_pipeline_model_package(model, calib, tmp_dir, metadata={"test": True})
        loaded_model, loaded_calib, meta = load_pipeline_model_package(save_path)

        assert loaded_model is not None
        assert meta.get("model_version") == "astra_pipeline_xgb_v1"
