"""
test_end_to_end_scoring.py
End-to-end integration test verifying candidate ranking from Stage 5 through Stage 11.
"""

import pytest
import numpy as np
import tempfile
import os

from astra_pipeline_scorer.src.utils import generate_synthetic_candidate_tree
from astra_pipeline_scorer.src.dataset_builder import PipelineDatasetBuilder
from astra_pipeline_scorer.src.training import train_pipeline_scorer
from astra_pipeline_scorer.src.checkpoint import save_pipeline_model_package
from astra_pipeline_scorer.src.inference import PipelineScoringEngine
from astra_pipeline_scorer.src.models import ConfidenceTier


def test_full_synthetic_pipeline_scoring_integration():
    """
    Generate multi-signal candidate trees, train XGBoost ranker, save checkpoint,
    and verify high Top-1 accuracy and confidence tiering on unseen test signals.
    """
    # 1. Generate 30 synthetic signal captures (each with 10 candidates)
    all_cands = []
    for s_idx in range(30):
        cands, _ = generate_synthetic_candidate_tree(
            f"sig_{s_idx:03d}",
            true_modulation="QPSK" if s_idx % 2 == 0 else "16QAM",
            true_symbol_rate_hz=9600.0 if s_idx % 3 == 0 else 4800.0,
            num_competing_candidates=8,
            seed=s_idx * 13,
        )
        all_cands.extend(cands)

    # 2. Build group-split dataset (70% train, 15% val, 15% test)
    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    val_sigs = [m["signal_id"] for m in data["val_meta"]]
    test_sigs = [m["signal_id"] for m in data["test_meta"]]

    # 3. Train XGBoost
    model, calib, val_rep, meta = train_pipeline_scorer(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        val_signal_ids=val_sigs,
    )

    assert val_rep.candidate_recall_top1 >= 0.70

    # 4. Save model checkpoint & initialize inference engine
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_pipeline_model_package(model, calib, tmp_dir)
        engine = PipelineScoringEngine(checkpoint_dir=tmp_dir)

        # 5. Test inference on unseen test signal
        test_cands, _ = generate_synthetic_candidate_tree("unseen_sig_01", num_competing_candidates=10, seed=999)
        result = engine.rank(test_cands, signal_id="unseen_sig_01", top_k=3)

        assert result.fallback_used is False
        assert len(result.ranked_candidates) == 3
        assert result.ranked_candidates[0].candidate_id == "cand_correct"
        assert result.top1_score > result.top2_score
        assert result.confidence_tier in (ConfidenceTier.CONFIRMED, ConfidenceTier.ESTIMATED)
