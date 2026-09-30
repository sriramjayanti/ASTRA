"""
ASTRA Symbol-Rate Ranker Training Pipeline.
Trains XGBoost on candidate evidence with GroupKFold splitting by source signal.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .candidate_features import FEATURE_COLUMNS
from .dataset_builder import (
    build_candidate_dataset_from_signals,
    generate_synthetic_training_pool,
)
from .xgboost_ranker import XGBoostSymbolRateRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("astra_symbol_rate.train")


def train_symbol_rate_ranker(
    num_signals: int = 140,
    test_size: float = 0.25,
    random_state: int = 42,
    output_dir: str = "checkpoints",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[XGBoostSymbolRateRanker, Dict[str, Any]]:
    """
    Executes the end-to-end training pipeline for the ASTRA symbol rate candidate ranker.
    """
    logger.info("Step 1: Generating synthetic training signal pool (%d signals)...", num_signals)
    signals = generate_synthetic_training_pool(num_signals=num_signals, seed=random_state)

    logger.info("Step 2: Extracting DSP evidence and building candidate dataset...")
    dataset = build_candidate_dataset_from_signals(signals, tolerance_percent=2.0, config=config)

    X = dataset["X"]
    y = dataset["y"]
    groups = dataset["groups"]
    candidate_recall = dataset["candidate_recall"]
    logger.info("Total candidate rows: %d (Positives: %d, Negatives: %d)", len(y), int(np.sum(y)), int(len(y) - np.sum(y)))
    logger.info("Overall Candidate Recall: %.2f%%", candidate_recall * 100.0)

    # Step 3: Group-level train/validation split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]

    # Verification of zero leakage
    train_unique_signals = set(groups_train)
    val_unique_signals = set(groups_val)
    overlap = train_unique_signals.intersection(val_unique_signals)
    if overlap:
        raise RuntimeError(f"FATAL: Group split leakage detected! Overlapping signals: {overlap}")
    logger.info("Verified group split: Train signals=%d, Val signals=%d (Zero overlap)", len(train_unique_signals), len(val_unique_signals))

    # Step 4: Train XGBoost Ranker
    ranker = XGBoostSymbolRateRanker(config=config)
    eval_metrics = ranker.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=FEATURE_COLUMNS,
    )
    logger.info("Training completed. Val ROC-AUC: %.4f | Val PR-AUC: %.4f | Log Loss: %.4f",
                eval_metrics.get("val_roc_auc", 0.0), eval_metrics.get("val_pr_auc", 0.0), eval_metrics.get("val_log_loss", 0.0))

    # Step 5: Save Model Checkpoint
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "symbol_rate_ranker.joblib")
    ranker.save(save_path)
    logger.info("Model saved successfully to %s", save_path)

    # Step 6: Feature Importances
    importances = ranker.get_feature_importances()
    logger.info("Top Feature Importances (Gain):")
    sorted_feats = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    for feat_name, gain_val in sorted_feats[:7]:
        logger.info("  - %-25s: %.4f", feat_name, gain_val)

    report = {
        "candidate_recall": candidate_recall,
        "eval_metrics": eval_metrics,
        "feature_importances": importances,
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "checkpoint_path": save_path,
    }
    return ranker, report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train ASTRA Symbol Rate Ranker")
    parser.add_argument("--num_signals", type=int, default=140, help="Number of synthetic signals")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()

    train_symbol_rate_ranker(num_signals=args.num_signals, output_dir=args.output_dir)
