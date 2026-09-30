"""
score_pipeline_candidates.py
Demonstration script showing complete candidate pipeline ranking and feature importance using Stage 11 XGBoost.
"""

import os
import sys
import json
import numpy as np

# Add workspace root to sys.path for standalone execution
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from astra_pipeline_scorer.src.utils import generate_synthetic_candidate_tree
from astra_pipeline_scorer.src.dataset_builder import PipelineDatasetBuilder
from astra_pipeline_scorer.src.training import train_pipeline_scorer
from astra_pipeline_scorer.src.checkpoint import save_pipeline_model_package
from astra_pipeline_scorer.src.inference import PipelineScoringEngine


def main():
    print("=" * 75)
    print("ASTRA STAGE 11 — PIPELINE SCORING MODEL (XGBOOST) DEMONSTRATION")
    print("=" * 75)

    # 1. Generate Synthetic Training Data
    print("\n[1] Generating multi-signal candidate trees (25 signals x 10 candidates)...")
    all_cands = []
    for s_idx in range(25):
        cands, _ = generate_synthetic_candidate_tree(
            f"sig_{s_idx:03d}",
            true_modulation="QPSK" if s_idx % 2 == 0 else "16QAM",
            true_symbol_rate_hz=9600.0 if s_idx % 3 == 0 else 4800.0,
            num_competing_candidates=10,
            seed=s_idx * 17,
        )
        all_cands.extend(cands)

    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, train_ratio=0.75, val_ratio=0.25, test_ratio=0.0, seed=42)
    val_sigs = [m["signal_id"] for m in data["val_meta"]]

    # 2. Train XGBoost Ranker & Calibrator
    print(f"\n[2] Training XGBoost model on {len(data['X_train'])} candidate paths...")
    model, calib, val_report, meta = train_pipeline_scorer(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        val_signal_ids=val_sigs,
    )
    print(f"    Validation Top-1 Accuracy : {val_report.candidate_recall_top1 * 100:.1f}%")
    print(f"    Validation ROC-AUC        : {val_report.roc_auc:.4f}")
    print(f"    Validation MRR            : {val_report.mrr:.4f}")

    # 3. Save model checkpoint & initialize inference engine
    checkpoint_dir = os.path.join(WORKSPACE_ROOT, "checkpoints", "pipeline_scorer_xgb")
    save_pipeline_model_package(model, calib, checkpoint_dir)
    print(f"\n[3] Model package saved to: {checkpoint_dir}")

    engine = PipelineScoringEngine(checkpoint_dir=checkpoint_dir)

    # 4. Rank candidates for a new unseen capture
    print("\n[4] Generating unseen test capture candidate tree...")
    test_cands, true_gt = generate_synthetic_candidate_tree(
        "capture_rf_042",
        true_modulation="QPSK",
        true_symbol_rate_hz=9600.0,
        true_interleaver="block",
        true_fec="convolutional",
        num_competing_candidates=10,
        seed=101,
    )

    ranking_result = engine.rank(test_cands, signal_id="capture_rf_042", top_k=3)

    # 5. Display Final Ranking Report
    print("\n" + "=" * 75)
    print("PIPELINE SCORER RANKING RESULT")
    print("=" * 75)
    print(f"Signal ID           : {ranking_result.signal_id}")
    print(f"Confidence Tier     : {ranking_result.confidence_tier.value}")
    print(f"Top-1 Score         : {ranking_result.top1_score:.4f}")
    print(f"Score Margin        : {ranking_result.score_margin:.4f} (Top-1 vs Top-2)")
    print(f"Ranking Uncertainty : {ranking_result.ranking_uncertainty:.4f}")
    print(f"Model Version       : {ranking_result.model_version}")

    print("\nRANKED CANDIDATE PATHS (TOP-3):")
    for cand in ranking_result.ranked_candidates:
        print(f"\n  [Rank #{cand.rank}] Path ID: {cand.pipeline_path_id}")
        print(f"    Score / Calibrated Prob : {cand.pipeline_score:.4f} / {cand.calibrated_probability:.4f}")
        print(f"    Modulation / Baud       : {cand.modulation} @ {cand.symbol_rate_hz:.0f} Hz ({cand.phase_variant})")
        print(f"    Interleaver / FEC       : {cand.interleaver} / {cand.fec}")
        print(f"    Validation Status       : {cand.validation_status}")
        if cand.key_evidence.get("top_positive_evidence"):
            print(f"    Key Drivers (+)         : {', '.join(cand.key_evidence['top_positive_evidence'])}")
        if cand.key_evidence.get("top_negative_evidence"):
            print(f"    Key Drivers (-)         : {', '.join(cand.key_evidence['top_negative_evidence'])}")

    print("\n" + "=" * 75)
    print("JSON EXPORT:")
    print("=" * 75)
    print(json.dumps(ranking_result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
