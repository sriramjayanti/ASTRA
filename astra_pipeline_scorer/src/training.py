"""
training.py
XGBoost training pipeline for ASTRA Stage 11 — Pipeline Scoring Model.
Handles class imbalance weighting, early stopping on validation splits, and probability calibration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from .feature_schema import PIPELINE_FEATURE_COLUMNS
from .calibration import ScoreCalibrator
from .evaluation import evaluate_pipeline_model
from .models import EvaluationReport


def train_pipeline_scorer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_signal_ids: Optional[List[str]] = None,
    val_signal_ids: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[xgb.XGBClassifier, ScoreCalibrator, EvaluationReport, Dict[str, Any]]:
    """
    Train an XGBoost classifier with early stopping and isotonic probability calibration.
    """
    if feature_names is None:
        feature_names = list(PIPELINE_FEATURE_COLUMNS)

    config = config or {}
    xgb_cfg = config.get("xgboost", {})

    # Calculate scale_pos_weight if not specified
    n_pos = max(1, int(np.sum(y_train == 1)))
    n_neg = max(1, int(np.sum(y_train == 0)))
    auto_scale = float(n_neg) / float(n_pos)

    # Initialize XGBClassifier
    model = xgb.XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 300)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        min_child_weight=float(xgb_cfg.get("min_child_weight", 2.0)),
        reg_alpha=float(xgb_cfg.get("reg_alpha", 0.1)),
        reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
        scale_pos_weight=auto_scale,
        random_state=int(xgb_cfg.get("random_state", 42)),
        tree_method=xgb_cfg.get("tree_method", "hist"),
        eval_metric=xgb_cfg.get("eval_metric", "logloss"),
        early_stopping_rounds=int(xgb_cfg.get("early_stopping_rounds", 30)),
    )

    # Fit with early stopping on validation set
    if len(X_val) > 0 and len(np.unique(y_val)) > 1:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    # Raw validation predictions
    raw_val_preds = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])

    # Probability calibration on validation set
    calibrator = ScoreCalibrator(method=config.get("calibration", {}).get("method", "isotonic"))
    if len(X_val) > 0 and len(np.unique(y_val)) > 1:
        calibrator.fit(raw_val_preds, y_val)
        calibrated_val_preds = calibrator.calibrate(raw_val_preds)
    else:
        calibrated_val_preds = raw_val_preds

    # Evaluate validation metrics
    val_report = EvaluationReport()
    if len(X_val) > 0 and val_signal_ids:
        val_report = evaluate_pipeline_model(
            val_signal_ids,
            raw_val_preds,
            y_val,
            calibrated_val_preds,
        )

    meta = {
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
        "scale_pos_weight": round(auto_scale, 2),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }

    return model, calibrator, val_report, meta
