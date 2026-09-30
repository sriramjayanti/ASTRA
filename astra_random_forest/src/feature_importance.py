"""
ASTRA Random Forest Feature Importance and Explainability Generator.
Computes Gini Impurity and Permutation Importance across all 36 DSP features.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.inspection import permutation_importance

from .family_classifier import BroadFamilyClassifier
from .feature_schema import FEATURE_COLUMNS


def compute_comprehensive_feature_importance(
    model: BroadFamilyClassifier,
    X_val: np.ndarray,
    y_val: Sequence[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Computes both Gini impurity and out-of-sample Permutation Importance.
    Returns a list of feature importance dicts sorted descending by Gini importance.
    """
    gini_dict = model.get_feature_importances()

    # Compute Permutation Importance
    perm_means: Dict[str, float] = {}
    perm_stds: Dict[str, float] = {}

    if len(X_val) > 0 and model.is_fitted:
        try:
            perm_res = permutation_importance(
                model.model,
                X_val,
                y_val,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1,
            )
            for i, col in enumerate(FEATURE_COLUMNS):
                perm_means[col] = float(perm_res.importances_mean[i])
                perm_stds[col] = float(perm_res.importances_std[i])
        except Exception:
            perm_means = {col: 0.0 for col in FEATURE_COLUMNS}
            perm_stds = {col: 0.0 for col in FEATURE_COLUMNS}
    else:
        perm_means = {col: 0.0 for col in FEATURE_COLUMNS}
        perm_stds = {col: 0.0 for col in FEATURE_COLUMNS}

    importance_list = []
    for col in FEATURE_COLUMNS:
        importance_list.append({
            "feature": col,
            "gini_importance": gini_dict.get(col, 0.0),
            "permutation_importance_mean": perm_means.get(col, 0.0),
            "permutation_importance_std": perm_stds.get(col, 0.0),
        })

    # Sort descending by Gini Importance
    importance_list.sort(key=lambda item: item["gini_importance"], reverse=True)

    # Assign ranks
    for rank, item in enumerate(importance_list, start=1):
        item["rank"] = rank

    return importance_list


def export_feature_importance_csv(
    importance_records: Sequence[Dict[str, Any]],
    output_csv_path: str,
) -> None:
    """
    Exports feature importance records to CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    fields = ["rank", "feature", "gini_importance", "permutation_importance_mean", "permutation_importance_std"]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in importance_records:
            writer.writerow({
                "rank": rec["rank"],
                "feature": rec["feature"],
                "gini_importance": f"{rec['gini_importance']:.6f}",
                "permutation_importance_mean": f"{rec['permutation_importance_mean']:.6f}",
                "permutation_importance_std": f"{rec['permutation_importance_std']:.6f}",
            })
