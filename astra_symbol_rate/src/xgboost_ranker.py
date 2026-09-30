"""
ASTRA XGBoost Symbol-Rate Candidate Ranker.
Ranks proposed candidate symbol rates using engineered DSP evidence and features.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from .candidate_features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from .models import ModelNotFittedError, SymbolRateCandidate


class XGBoostSymbolRateRanker:
    """
    Candidate correctness ranker model based on Gradient-Boosted Decision Trees (XGBoost).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.80,
        colsample_bytree: float = 0.80,
        scale_pos_weight: float = 3.0,
        random_state: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.feature_columns = list(FEATURE_COLUMNS)
        self.feature_schema_version = FEATURE_SCHEMA_VERSION
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {}
        self.config = config or {}

        xgb_cfg = self.config.get("symbol_rate", {}).get("xgboost", {})
        if isinstance(xgb_cfg, dict):
            n_estimators = xgb_cfg.get("n_estimators", n_estimators)
            max_depth = xgb_cfg.get("max_depth", max_depth)
            learning_rate = xgb_cfg.get("learning_rate", learning_rate)


        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                eval_metric="logloss",
                tree_method="hist",
            )
            self.backend = "xgboost"
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            self.backend = "sklearn_hist_gb"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> XGBoostSymbolRateRanker:
        """
        Trains candidate correctness binary classifier.
        """
        if self.backend == "xgboost" and eval_set is not None:
            self.model.fit(
                X, y,
                eval_set=[eval_set],
                verbose=verbose,
            )
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Unified training interface with validation metrics computation.
        """
        if feature_names:
            self.feature_columns = list(feature_names)

        eval_set = (X_val, y_val) if (X_val is not None and y_val is not None) else None
        self.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)

        metrics = {}
        if X_val is not None and y_val is not None and len(y_val) > 0:
            probs = self.predict_proba(X_val)
            from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
            try:
                metrics["val_roc_auc"] = float(roc_auc_score(y_val, probs))
            except Exception:
                metrics["val_roc_auc"] = 0.5
            try:
                metrics["val_pr_auc"] = float(average_precision_score(y_val, probs))
            except Exception:
                metrics["val_pr_auc"] = float(np.mean(y_val))
            try:
                metrics["val_log_loss"] = float(log_loss(y_val, probs))
            except Exception:
                metrics["val_log_loss"] = 0.0

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns P(candidate is correct) for each candidate row.
        """
        if not self.is_fitted:
            # If not fitted yet, fallback to heuristic baseline using support count and scores
            if len(X) == 0:
                return np.zeros(0, dtype=np.float32)
            # Baseline DSP weighted score
            scores = (
                X[:, 3] * 0.25 +  # autocorr_score
                X[:, 4] * 0.25 +  # power_autocorr_score
                X[:, 5] * 0.25 +  # cyclostationary_score
                X[:, 6] * 0.15 +  # bandwidth_consistency
                X[:, 10] * 0.10   # support_count / 10
            )
            probs = np.clip(scores, 0.01, 0.99)
            return probs.astype(np.float32)

        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)

        probs_all = self.model.predict_proba(X)
        if probs_all.ndim == 2:
            return probs_all[:, 1].astype(np.float32)
        return probs_all.astype(np.float32)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_proba."""
        return self.predict_proba(X)

    def rank_candidates(
        self,
        candidates: Sequence[SymbolRateCandidate],
        feature_matrix: np.ndarray,
    ) -> List[SymbolRateCandidate]:
        """
        Scores and sorts candidate hypotheses in descending order of correctness likelihood.
        """
        if not candidates or len(feature_matrix) == 0:
            return list(candidates)

        scores = self.predict_proba(feature_matrix)
        for cand, sc in zip(candidates, scores):
            cand.score = float(sc)

        # Sort descending by score
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importance mapping if supported."""
        if not self.is_fitted:
            return {col: 0.0 for col in self.feature_columns}

        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
            return {col: float(imps[i]) for i, col in enumerate(self.feature_columns)}
        return {col: 1.0 / len(self.feature_columns) for col in self.feature_columns}

    def save(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Alias for save_checkpoint."""
        self.save_checkpoint(path, extra_meta)

    def load(self, path: str) -> None:
        """Loads weights from checkpoint into self."""
        loaded = self.load_checkpoint(path)
        self.model = loaded.model
        self.backend = loaded.backend
        self.feature_columns = loaded.feature_columns
        self.feature_schema_version = loaded.feature_schema_version
        self.is_fitted = loaded.is_fitted
        self.metadata = loaded.metadata

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves XGBoost model, feature schema version, and metadata bundle.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        import joblib
        bundle = {
            "model": self.model,
            "backend": self.backend,
            "feature_columns": self.feature_columns,
            "feature_schema_version": self.feature_schema_version,
            "is_fitted": self.is_fitted,
            "metadata": extra_meta or {},
        }
        joblib.dump(bundle, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> XGBoostSymbolRateRanker:
        """
        Loads ranker bundle and verifies feature schema version.
        """
        import joblib
        bundle = joblib.load(path)
        ranker = cls()
        ranker.model = bundle["model"]
        ranker.backend = bundle.get("backend", "xgboost")
        ranker.feature_columns = bundle["feature_columns"]
        ranker.feature_schema_version = bundle["feature_schema_version"]
        ranker.is_fitted = bundle["is_fitted"]
        ranker.metadata = bundle.get("metadata", {})

        if ranker.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Feature schema version mismatch: checkpoint has {ranker.feature_schema_version}, "
                f"runtime expects {FEATURE_SCHEMA_VERSION}"
            )
        return ranker

