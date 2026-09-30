"""
ASTRA Production Symbol-Rate (Baud) Estimation Engine Inference API.
Provides single-signal, batch, and chunked streaming estimation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import yaml

from .candidate_features import build_candidate_feature_matrix
from .candidate_generator import generate_candidates
from .confidence import compute_confidence_and_status
from .dataset_builder import extract_dsp_evidence
from .models import (
    DSPRateEvidence,
    InvalidSignalError,
    NoCandidatesFoundError,
    SymbolRateCandidate,
    SymbolRatePrediction,
)
from .preprocessing import preprocess_iq
from .xgboost_ranker import XGBoostSymbolRateRanker

logger = logging.getLogger("astra_symbol_rate.inference")


class SymbolRateEstimator:
    """
    Main ASTRA Symbol-Rate Estimation Engine class.
    Combines multi-modal DSP evidence extraction with XGBoost candidate ranking.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        top_k: int = 3,
    ) -> None:
        self.top_k = top_k
        self.config: Dict[str, Any] = {}
        self.model_version = "astra_symbol_rate_xgb_v1.0"

        # 1. Load config if provided or search default
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            default_cfg = Path(__file__).resolve().parent.parent / "configs" / "symbol_rate_config.yaml"
            if default_cfg.exists():
                with open(default_cfg, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}

        # 2. Load XGBoost model if available
        self.ranker = XGBoostSymbolRateRanker(config=self.config)
        resolved_model_path = model_path
        if resolved_model_path is None:
            default_ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / "symbol_rate_ranker.joblib"
            if default_ckpt.exists():
                resolved_model_path = str(default_ckpt)

        if resolved_model_path is not None and os.path.exists(resolved_model_path):
            try:
                self.ranker.load(resolved_model_path)
                logger.info("Loaded XGBoost symbol rate ranker from: %s", resolved_model_path)
            except Exception as e:
                logger.warning("Failed to load model from %s: %s. Using DSP ranking fallback.", resolved_model_path, e)
        else:
            logger.info("No trained checkpoint found. Estimator will operate in DSP voting fallback mode.")

    def estimate(
        self,
        iq: np.ndarray,
        sample_rate_hz: Optional[float] = None,
        modulation_prediction: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SymbolRatePrediction:
        """
        Estimates the symbol rate (baud) of an incoming IQ signal burst.
        Accepts flexible parameter conventions: sample_rate_hz, sample_rate, fs, sr, modulation.
        """
        # 1. Validation & Preprocessing
        if iq is None or len(iq) == 0:
            raise InvalidSignalError("Input IQ signal is empty or None.")
        if not np.all(np.isfinite(iq)):
            raise InvalidSignalError("Input IQ signal contains NaN or infinite values.")

        actual_fs = sample_rate_hz
        if actual_fs is None:
            actual_fs = kwargs.get("sample_rate") or kwargs.get("fs") or kwargs.get("sr") or kwargs.get("input_sample_rate_hz")
        if actual_fs is None or actual_fs <= 0:
            raise InvalidSignalError(f"Invalid sample rate: {actual_fs}")
        actual_fs = float(actual_fs)

        mod_hint = kwargs.get("modulation") or kwargs.get("modulation_hint")
        if mod_hint is None and modulation_prediction:
            mod_hint = modulation_prediction.get("predicted_modulation") or modulation_prediction.get("modulation")

        # Extract DSP Evidence
        evidence = extract_dsp_evidence(iq, actual_fs, self.config)

        # Generate Candidate Hypotheses
        candidates = generate_candidates(evidence, actual_fs, self.config, modulation_hint=mod_hint)

        if not candidates:
            # Return UNKNOWN prediction gracefully
            return SymbolRatePrediction(
                best_symbol_rate_hz=0.0,
                samples_per_symbol=0.0,
                confidence=0.0,
                status="UNKNOWN",
                confidence_margin=0.0,
                top_k=[],
                dsp_evidence=evidence.to_dict(),
                model_version=self.model_version,
                unknown_reason="No plausible DSP candidate rates identified.",
                modulation_context=modulation_prediction,
            )

        # Build Candidate Feature Matrix
        feat_matrix, _ = build_candidate_feature_matrix(
            candidates=candidates,
            evidence=evidence,
            sample_rate_hz=actual_fs,
            modulation_context=modulation_prediction,
        )

        # Score Candidates with physics-guided ranking
        # Preserves physical cyclostationary / transition evidence without artificial frequency bias
        if self.ranker.is_fitted and len(feat_matrix) > 0:
            try:
                ml_scores = self.ranker.predict_scores(feat_matrix)
                for cand, ml_s in zip(candidates, ml_scores):
                    # Blend physical evidence score (70%) with ML score (30%)
                    cand.score = float(0.70 * cand.score + 0.30 * float(ml_s) * 10.0)
            except Exception as e:
                logger.debug("Ranker scoring fallback: %s", e)

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

        # Build Top-K Output
        top_k_candidates = ranked[: self.top_k]
        top_k_dict = []
        for rank_idx, cand in enumerate(top_k_candidates, start=1):
            top_k_dict.append({
                "rank": rank_idx,
                "rate_hz": float(cand.rate_hz),
                "samples_per_symbol": float(cand.samples_per_symbol),
                "score": float(round(cand.score, 4)),
                "supported_by": cand.supported_by,
                "sources": cand.sources,
                "harmonic_ratio": cand.harmonic_ratio,
            })

        best_cand = top_k_candidates[0]
        conf, status, margin, unk_reason = compute_confidence_and_status(
            candidates=ranked,
            evidence=evidence,
            config=self.config,
        )

        return SymbolRatePrediction(
            best_symbol_rate_hz=float(best_cand.rate_hz),
            samples_per_symbol=float(best_cand.samples_per_symbol),
            confidence=float(round(conf, 4)),
            status=status,
            confidence_margin=float(round(margin, 4)),
            top_k=top_k_dict,
            dsp_evidence=evidence.to_dict(),
            model_version=self.model_version,
            unknown_reason=unk_reason,
            modulation_context=modulation_prediction,
        )

    def estimate_batch(
        self,
        iq_batch: Sequence[np.ndarray],
        sample_rate_hz_list: Sequence[float],
        modulation_predictions: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[SymbolRatePrediction]:
        """
        Processes a batch of IQ signals.
        """
        if len(iq_batch) != len(sample_rate_hz_list):
            raise ValueError("iq_batch and sample_rate_hz_list must have identical length.")

        mod_preds = modulation_predictions or [None] * len(iq_batch)
        results = []
        for iq, fs, mod_ctx in zip(iq_batch, sample_rate_hz_list, mod_preds):
            try:
                res = self.estimate(iq, fs, modulation_prediction=mod_ctx)
                results.append(res)
            except Exception as e:
                logger.error("Error estimating batch signal: %s", e)
                results.append(SymbolRatePrediction(
                    best_symbol_rate_hz=0.0,
                    samples_per_symbol=0.0,
                    confidence=0.0,
                    status="UNKNOWN",
                    confidence_margin=0.0,
                    top_k=[],
                    dsp_evidence={},
                    unknown_reason=str(e),
                ))
        return results

    def estimate_stream(
        self,
        iq_stream: np.ndarray,
        sample_rate_hz: float,
        window_size: int = 2048,
        step_size: int = 1024,
        modulation_prediction: Optional[Dict[str, Any]] = None,
    ) -> SymbolRatePrediction:
        """
        Performs chunked window estimation over long captures with stability voting.
        """
        if len(iq_stream) < window_size:
            return self.estimate(iq_stream, sample_rate_hz, modulation_prediction)

        window_predictions: List[SymbolRatePrediction] = []
        for start_idx in range(0, len(iq_stream) - window_size + 1, step_size):
            chunk = iq_stream[start_idx : start_idx + window_size]
            pred = self.estimate(chunk, sample_rate_hz, modulation_prediction)
            if pred.status != "UNKNOWN":
                window_predictions.append(pred)

        if not window_predictions:
            return self.estimate(iq_stream[:window_size], sample_rate_hz, modulation_prediction)

        # Aggregate rates across windows using median
        rates = [p.best_symbol_rate_hz for p in window_predictions]
        med_rate = float(np.median(rates))
        
        # Stability score = fraction of windows within 5% of median
        close_windows = [r for r in rates if abs(r - med_rate) / max(1.0, med_rate) <= 0.05]
        stability = len(close_windows) / max(1, len(rates))

        # Select the prediction closest to median
        best_pred = min(window_predictions, key=lambda p: abs(p.best_symbol_rate_hz - med_rate))
        best_pred.confidence = float(np.clip(best_pred.confidence * (0.5 + 0.5 * stability), 0.0, 1.0))
        if stability < 0.4:
            best_pred.status = "POSSIBLE"

        return best_pred


class SymbolRateInferenceEngine:
    """
    Unified Inference Engine wrapper providing compatibility for ASTRA GUI, CLI, and pipelines.
    Adapts SymbolRateEstimator to expected predict(iq, sample_rate) interface.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        top_k: int = 3,
    ) -> None:
        self.estimator = SymbolRateEstimator(
            model_path=model_path,
            config_path=config_path,
            top_k=top_k,
        )

    def estimate(
        self,
        iq: np.ndarray,
        sample_rate_hz: Optional[float] = None,
        sample_rate: Optional[float] = None,
        modulation_prediction: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SymbolRatePrediction:
        """Estimates symbol rate, accepting both sample_rate and sample_rate_hz."""
        fs = sample_rate_hz if sample_rate_hz is not None else sample_rate
        return self.estimator.estimate(
            iq=iq,
            sample_rate_hz=fs,
            modulation_prediction=modulation_prediction,
            **kwargs,
        )

    def predict(
        self,
        iq: np.ndarray,
        sample_rate: float,
        modulation_prediction: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SymbolRatePrediction:
        """Alias for estimate() matching GUI and CLI pipeline calls."""
        return self.estimate(
            iq=iq,
            sample_rate=sample_rate,
            modulation_prediction=modulation_prediction,
            **kwargs,
        )

    def __call__(
        self,
        iq: np.ndarray,
        sample_rate: float,
        **kwargs: Any,
    ) -> SymbolRatePrediction:
        return self.estimate(iq=iq, sample_rate=sample_rate, **kwargs)

