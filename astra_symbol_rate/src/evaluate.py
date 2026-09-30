"""
ASTRA Symbol-Rate Comprehensive Evaluation Suite.
Evaluates accuracy vs SNR, Modulation, Baud Rate, and compares against DSP baselines.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .dataset_builder import extract_dsp_evidence
from .candidate_generator import generate_candidates
from .candidate_features import build_candidate_feature_matrix
from .confidence import compute_confidence_and_status
from .models import SymbolRatePrediction
from .utils import generate_synthetic_signal
from .xgboost_ranker import XGBoostSymbolRateRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("astra_symbol_rate.evaluate")


def evaluate_signal(
    iq: np.ndarray,
    sample_rate_hz: float,
    true_rate_hz: float,
    ranker: Optional[XGBoostSymbolRateRanker] = None,
    modulation_context: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Evaluates a single signal under DSP-only voting, Autocorr baseline, and XGBoost ranking.
    """
    evidence = extract_dsp_evidence(iq, sample_rate_hz)
    candidates = generate_candidates(evidence, sample_rate_hz)

    if not candidates:
        return {
            "true_rate_hz": true_rate_hz,
            "xgb_top1_rate": 0.0,
            "xgb_top1_error_pct": 100.0,
            "top1_correct_2pct": False,
            "top1_correct_5pct": False,
            "top3_correct_2pct": False,
            "dsp_vote_top1_rate": 0.0,
            "dsp_vote_error_pct": 100.0,
            "autocorr_top1_rate": 0.0,
            "autocorr_error_pct": 100.0,
        }

    # 1. Autocorr baseline
    autocorr_top1 = 0.0
    if evidence.autocorr_peaks:
        autocorr_top1 = evidence.autocorr_peaks[0]["rate_hz"]
    autocorr_err_pct = abs(autocorr_top1 - true_rate_hz) / max(1.0, true_rate_hz) * 100.0

    # 2. DSP Multi-source Voting baseline (rank by support_count then autocorr/cyclo score)
    sorted_by_dsp = sorted(
        candidates,
        key=lambda c: (len(c.supported_by), c.score),
        reverse=True,
    )
    dsp_vote_top1 = sorted_by_dsp[0].rate_hz
    dsp_vote_err_pct = abs(dsp_vote_top1 - true_rate_hz) / max(1.0, true_rate_hz) * 100.0

    # 3. XGBoost Ranking
    if ranker is not None and ranker.is_fitted:
        feat_matrix, _ = build_candidate_feature_matrix(
            candidates=candidates,
            evidence=evidence,
            sample_rate_hz=sample_rate_hz,
            modulation_context=modulation_context,
        )
        scores = ranker.predict_scores(feat_matrix)
        for cand, sc in zip(candidates, scores):
            cand.score = float(sc)
        ranked_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    else:
        ranked_candidates = sorted_by_dsp

    xgb_top1 = ranked_candidates[0].rate_hz
    xgb_top1_err_pct = abs(xgb_top1 - true_rate_hz) / max(1.0, true_rate_hz) * 100.0

    # Check Top-K contains correct rate
    top3_cands = ranked_candidates[:top_k]
    top3_correct_2pct = any(
        (abs(c.rate_hz - true_rate_hz) / max(1.0, true_rate_hz) <= 0.02)
        for c in top3_cands
    )
    top1_correct_2pct = (xgb_top1_err_pct <= 2.0)
    top1_correct_5pct = (xgb_top1_err_pct <= 5.0)
    top1_correct_1pct = (xgb_top1_err_pct <= 1.0)

    return {
        "true_rate_hz": true_rate_hz,
        "xgb_top1_rate": xgb_top1,
        "xgb_top1_error_pct": xgb_top1_err_pct,
        "abs_error_hz": abs(xgb_top1 - true_rate_hz),
        "top1_correct_1pct": top1_correct_1pct,
        "top1_correct_2pct": top1_correct_2pct,
        "top1_correct_5pct": top1_correct_5pct,
        "top3_correct_2pct": top3_correct_2pct,
        "dsp_vote_top1_rate": dsp_vote_top1,
        "dsp_vote_error_pct": dsp_vote_err_pct,
        "autocorr_top1_rate": autocorr_top1,
        "autocorr_error_pct": autocorr_err_pct,
    }


def run_benchmark_evaluation(
    ranker: Optional[XGBoostSymbolRateRanker] = None,
    num_test_signals: int = 100,
    seed: int = 999,
) -> Dict[str, Any]:
    """
    Executes a comprehensive evaluation over a test grid of signals.
    """
    np.random.seed(seed)
    mods = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM"]
    rates = [1200, 2400, 4800, 9600, 19200, 38400, 7350, 14200]
    snrs = [-5, 0, 5, 10, 15, 20]
    sample_rates = [96000, 192000, 250000]

    results: List[Dict[str, Any]] = []

    for i in range(num_test_signals):
        mod = mods[i % len(mods)]
        rate = rates[(i // len(mods)) % len(rates)]
        fs = sample_rates[(i // (len(mods) * len(rates))) % len(sample_rates)]
        while fs / rate < 2.5:
            fs *= 2

        snr = float(np.random.choice(snrs))
        cfo = float(np.random.uniform(-1000, 1000))

        iq, meta = generate_synthetic_signal(
            mod_type=mod,
            symbol_rate_hz=rate,
            sample_rate_hz=fs,
            num_symbols=1024,
            snr_db=snr,
            cfo_hz=cfo,
            seed=seed + i,
        )

        res = evaluate_signal(iq, fs, rate, ranker=ranker)
        res["mod_type"] = mod
        res["snr_db"] = snr
        res["sample_rate_hz"] = fs
        results.append(res)

    # 1. Global Metrics
    abs_errors = [r["abs_error_hz"] for r in results]
    rel_errors = [r["xgb_top1_error_pct"] for r in results]
    top1_2pct_acc = np.mean([1.0 if r["top1_correct_2pct"] else 0.0 for r in results]) * 100.0
    top1_5pct_acc = np.mean([1.0 if r["top1_correct_5pct"] else 0.0 for r in results]) * 100.0
    top3_2pct_acc = np.mean([1.0 if r["top3_correct_2pct"] else 0.0 for r in results]) * 100.0

    dsp_vote_acc = np.mean([1.0 if r["dsp_vote_error_pct"] <= 2.0 else 0.0 for r in results]) * 100.0
    autocorr_acc = np.mean([1.0 if r["autocorr_error_pct"] <= 2.0 else 0.0 for r in results]) * 100.0

    # 2. Breakdown vs SNR
    snr_breakdown = {}
    for snr_val in sorted(list(set(r["snr_db"] for r in results))):
        subset = [r for r in results if r["snr_db"] == snr_val]
        snr_breakdown[snr_val] = {
            "count": len(subset),
            "top1_acc_2pct": float(np.mean([1.0 if r["top1_correct_2pct"] else 0.0 for r in subset]) * 100.0),
            "top3_acc_2pct": float(np.mean([1.0 if r["top3_correct_2pct"] else 0.0 for r in subset]) * 100.0),
            "mean_rel_error_pct": float(np.mean([r["xgb_top1_error_pct"] for r in subset])),
        }

    # 3. Breakdown vs Modulation
    mod_breakdown = {}
    for mod_val in sorted(list(set(r["mod_type"] for r in results))):
        subset = [r for r in results if r["mod_type"] == mod_val]
        mod_breakdown[mod_val] = {
            "count": len(subset),
            "top1_acc_2pct": float(np.mean([1.0 if r["top1_correct_2pct"] else 0.0 for r in subset]) * 100.0),
            "top3_acc_2pct": float(np.mean([1.0 if r["top3_correct_2pct"] else 0.0 for r in subset]) * 100.0),
            "mean_rel_error_pct": float(np.mean([r["xgb_top1_error_pct"] for r in subset])),
        }

    summary = {
        "num_test_signals": len(results),
        "mean_abs_error_hz": float(np.mean(abs_errors)),
        "median_abs_error_hz": float(np.median(abs_errors)),
        "mean_relative_error_pct": float(np.mean(rel_errors)),
        "top1_accuracy_2pct": top1_2pct_acc,
        "top1_accuracy_5pct": top1_5pct_acc,
        "top3_accuracy_2pct": top3_2pct_acc,
        "dsp_vote_accuracy_2pct": dsp_vote_acc,
        "autocorr_baseline_accuracy_2pct": autocorr_acc,
        "snr_breakdown": snr_breakdown,
        "mod_breakdown": mod_breakdown,
    }
    return summary
