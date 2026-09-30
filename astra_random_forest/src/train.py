"""
ASTRA Random Forest Support Model Training Pipeline.
Trains Broad Family and Signal Quality Random Forests with GroupKFold leakage-free splitting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
import yaml

from .checkpoint import save_random_forest_bundle
from .dataset_builder import build_dataset_from_signals, generate_synthetic_rf_pool
from .family_classifier import BroadFamilyClassifier
from .feature_importance import compute_comprehensive_feature_importance, export_feature_importance_csv
from .feature_schema import FEATURE_COLUMNS
from .preprocessing import FeaturePreprocessor, verify_no_label_leakage
from .quality_classifier import SignalQualityClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("astra_random_forest.train")


def train_random_forest_support(
    num_signals: int = 160,
    test_size: float = 0.25,
    random_state: int = 42,
    output_dir: str = "checkpoints",
    config_path: Optional[str] = None,
) -> Tuple[BroadFamilyClassifier, SignalQualityClassifier, Dict[str, Any]]:
    """
    End-to-end training pipeline for Random Forest family and quality models.
    """
    logger.info("Step 1: Generating training signal pool (%d signals)...", num_signals)
    signals = generate_synthetic_rf_pool(num_signals=num_signals, seed=random_state)

    logger.info("Step 2: Extracting 36 DSP features and quality truth labels...")
    dataset = build_dataset_from_signals(signals)

    X = dataset["X"]
    y_family = dataset["y_family"]
    y_quality = dataset["y_quality"]
    groups = dataset["groups"]

    verify_no_label_leakage(X, FEATURE_COLUMNS)

    # Step 3: Leakage-Free Group Partitioning
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y_family, groups=groups))

    X_train, y_fam_train, y_qual_train, groups_train = X[train_idx], y_family[train_idx], y_quality[train_idx], groups[train_idx]
    X_val, y_fam_val, y_qual_val, groups_val = X[val_idx], y_family[val_idx], y_quality[val_idx], groups[val_idx]

    # Verify zero group leakage
    overlap = set(groups_train).intersection(set(groups_val))
    if overlap:
        raise RuntimeError(f"FATAL: Group leakage detected across train/val split! Overlapping: {overlap}")
    logger.info("Verified group split: Train signals=%d, Val signals=%d (Zero overlap)", len(set(groups_train)), len(set(groups_val)))

    # Step 4: Fit Feature Imputation Pipeline
    preprocessor = FeaturePreprocessor(strategy="median")
    X_train_imp = preprocessor.fit_transform(X_train)
    X_val_imp = preprocessor.transform(X_val)

    # Step 5: Train Broad Family Classifier
    logger.info("Step 3: Training Broad Family Classifier (FSK/PSK/QAM/UNKNOWN)...")
    family_model = BroadFamilyClassifier(
        n_estimators=300,
        max_depth=14,
        class_weight="balanced",
        random_state=random_state,
    )
    family_model.fit(X_train_imp, y_fam_train)

    fam_preds = family_model.predict(X_val_imp)
    fam_acc = float(accuracy_score(y_fam_val, fam_preds))
    fam_bal_acc = float(balanced_accuracy_score(y_fam_val, fam_preds))
    fam_macro_f1 = float(f1_score(y_fam_val, fam_preds, average="macro", zero_division=0))
    fam_macro_prec = float(precision_score(y_fam_val, fam_preds, average="macro", zero_division=0))
    fam_macro_rec = float(recall_score(y_fam_val, fam_preds, average="macro", zero_division=0))

    logger.info("Family Model Val Accuracy: %.4f | Macro-F1: %.4f | Balanced Acc: %.4f", fam_acc, fam_macro_f1, fam_bal_acc)

    # Step 6: Train Signal Quality Classifier
    logger.info("Step 4: Training Multi-Label Signal Quality Classifier...")
    quality_model = SignalQualityClassifier(
        labels=dataset["quality_label_names"],
        n_estimators=200,
        max_depth=10,
        random_state=random_state,
    )
    quality_model.fit(X_train_imp, y_qual_train)

    qual_preds = quality_model.predict(X_val_imp)
    qual_metrics = {}
    for idx, lbl in enumerate(dataset["quality_label_names"]):
        lbl_acc = float(accuracy_score(y_qual_val[:, idx], qual_preds[:, idx]))
        lbl_f1 = float(f1_score(y_qual_val[:, idx], qual_preds[:, idx], zero_division=0))
        qual_metrics[lbl] = {"accuracy": lbl_acc, "f1": lbl_f1}
        logger.info("Quality [%-12s] Val Acc: %.4f | F1: %.4f", lbl, lbl_acc, lbl_f1)

    # Step 7: Feature Importance Analysis
    logger.info("Step 5: Evaluating Gini and Permutation Feature Importance...")
    importance_records = compute_comprehensive_feature_importance(
        model=family_model,
        X_val=X_val_imp,
        y_val=y_fam_val,
        n_repeats=5,
        random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "feature_importance.csv")
    export_feature_importance_csv(importance_records, csv_path)
    logger.info("Feature importance exported to: %s", csv_path)

    # Print Top 7 features
    logger.info("Top Feature Importances (Gini):")
    for rec in importance_records[:7]:
        logger.info("  #%d %-28s: Gini=%.4f, Perm=%.4f", rec["rank"], rec["feature"], rec["gini_importance"], rec["permutation_importance_mean"])

    metrics = {
        "family": {
            "accuracy": fam_acc,
            "balanced_accuracy": fam_bal_acc,
            "macro_f1": fam_macro_f1,
            "macro_precision": fam_macro_prec,
            "macro_recall": fam_macro_rec,
        },
        "quality": qual_metrics,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }

    # Step 8: Save Model Checkpoint
    ckpt_path = os.path.join(output_dir, "random_forest_support.joblib")
    save_random_forest_bundle(
        save_path=ckpt_path,
        family_model=family_model,
        quality_model=quality_model,
        preprocessor=preprocessor,
        metrics=metrics,
    )
    logger.info("Checkpoint saved to: %s", ckpt_path)

    return family_model, quality_model, metrics


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train ASTRA Random Forest Support Model")
    parser.add_argument("--num_signals", type=int, default=160, help="Number of synthetic signals")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()

    train_random_forest_support(num_signals=args.num_signals, output_dir=args.output_dir)
