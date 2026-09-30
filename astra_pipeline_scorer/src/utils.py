"""
utils.py
Synthetic multi-candidate tree generation and dataset creation utilities for Stage 11.
Generates realistic candidate trees per signal with known ground truth, hard negatives,
and diverse channel/demodulation parameters for model training and integration testing.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .models import PipelinePathCandidate
from .label_builder import evaluate_candidate_correctness


def generate_synthetic_candidate_tree(
    signal_id: str,
    true_modulation: str = "QPSK",
    true_symbol_rate_hz: float = 9600.0,
    true_interleaver: str = "block",
    true_fec: str = "convolutional",
    snr_db: float = 12.0,
    num_competing_candidates: int = 12,
    seed: Optional[int] = None,
) -> Tuple[List[PipelinePathCandidate], Dict[str, Any]]:
    """
    Generate a realistic candidate tree for one signal capture, including:
    1. True correct pipeline candidate (top demod/sync/CRC metrics)
    2. Hard negative candidates (correct modulation + wrong baud, correct baud + wrong interleaver, etc.)
    3. Easy negative candidates (wrong modulation, failed sync, failed CRC)
    """
    if seed is not None:
        np.random.seed(seed)

    ground_truth = {
        "modulation": true_modulation,
        "symbol_rate_hz": true_symbol_rate_hz,
        "phase_variant": "rot0",
        "interleaver_family": true_interleaver,
        "fec_family": true_fec,
    }

    candidates = []

    # 1. Correct Candidate
    cand_correct = PipelinePathCandidate(
        signal_id=signal_id,
        candidate_path_id=f"{signal_id}_path_correct",
        candidate_id="cand_correct",
        demod_variant_id="rot0",
        interleaver_candidate_id=true_interleaver,
        fec_candidate_id=true_fec,
        modulation=true_modulation,
        symbol_rate_hz=true_symbol_rate_hz,
        phase_variant="rot0",
        interleaver_family=true_interleaver,
        fec_family=true_fec,
        stage5_metrics={
            "modulation": true_modulation,
            "modulation_probability": 0.82 + np.random.uniform(-0.05, 0.05),
            "symbol_rate_score": 0.88 + np.random.uniform(-0.05, 0.05),
            "candidate_initial_score": 0.85,
            "rf_family_support": 0.90,
            "constellation_support": 0.85,
            "modulation_rank": 1.0,
            "baud_rank": 1.0,
            "samples_per_symbol": 4.0,
        },
        stage6_sync_metrics={
            "timing_lock_score": 0.92 + np.random.uniform(-0.03, 0.03),
            "carrier_lock_score": 0.90,
            "overall_sync_score": 0.91,
            "phase_error_rms": 0.05,
            "constellation_quality_after": 0.88,
        },
        stage7_demod_metrics={
            "evm_percent": max(5.0, 15.0 - snr_db * 0.5),
            "evm_db": -20.0,
            "decision_distance_mean": 0.85,
            "mean_abs_llr": 3.5,
            "estimated_snr_db": snr_db,
            "demod_quality_score": 0.90,
        },
        stage8_interleaver_metrics={
            "interleaver_family": true_interleaver,
            "interleaver_structural_score": 0.88,
            "interleaver_periodicity_score": 0.85,
        },
        stage9_fec_metrics={
            "fec_family": true_fec,
            "decoder_success": True,
            "fec_quality_score": 0.95,
            "normalized_path_metric": 0.02,
            "ldpc_converged": True,
        },
        stage10_validation_metrics={
            "crc_available": 1.0,
            "crc_pass": 1.0,
            "crc_width": 16.0,
            "crc_pass_rate": 1.0,
            "syndrome_valid": 1.0,
            "frame_periodicity_score": 0.80,
            "header_consistency_score": 1.0,
            "length_consistency_score": 1.0,
            "independent_evidence_groups": 5.0,
            "strong_evidence_groups": 3.0,
            "contradiction_count": 0.0,
            "overall_validation_score": 0.92,
            "validation_status": "VALIDATION_STRONG",
        },
    )
    cand_correct.candidate_correct = evaluate_candidate_correctness(cand_correct, ground_truth)
    candidates.append(cand_correct)

    # 2. Hard Negative: Correct Mod/Baud but Wrong Interleaver
    cand_wrong_inter = PipelinePathCandidate(
        signal_id=signal_id,
        candidate_path_id=f"{signal_id}_path_wrong_inter",
        candidate_id="cand_wrong_inter",
        modulation=true_modulation,
        symbol_rate_hz=true_symbol_rate_hz,
        interleaver_family="none" if true_interleaver != "none" else "block",
        fec_family=true_fec,
        stage5_metrics=dict(cand_correct.stage5_metrics),
        stage6_sync_metrics=dict(cand_correct.stage6_sync_metrics),
        stage7_demod_metrics=dict(cand_correct.stage7_demod_metrics),
        stage8_interleaver_metrics={"interleaver_structural_score": 0.30},
        stage9_fec_metrics={"decoder_success": False, "fec_quality_score": 0.20, "normalized_path_metric": 0.40},
        stage10_validation_metrics={
            "crc_available": 1.0,
            "crc_pass": 0.0,
            "crc_pass_rate": 0.0,
            "syndrome_valid": 0.0,
            "contradiction_count": 2.0,
            "overall_validation_score": 0.15,
            "validation_status": "VALIDATION_FAILED",
        },
    )
    cand_wrong_inter.candidate_correct = evaluate_candidate_correctness(cand_wrong_inter, ground_truth)
    candidates.append(cand_wrong_inter)

    # 3. Hard Negative: Correct Mod/Baud but Wrong FEC
    cand_wrong_fec = PipelinePathCandidate(
        signal_id=signal_id,
        candidate_path_id=f"{signal_id}_path_wrong_fec",
        candidate_id="cand_wrong_fec",
        modulation=true_modulation,
        symbol_rate_hz=true_symbol_rate_hz,
        interleaver_family=true_interleaver,
        fec_family="reed_solomon" if true_fec != "reed_solomon" else "convolutional",
        stage5_metrics=dict(cand_correct.stage5_metrics),
        stage6_sync_metrics=dict(cand_correct.stage6_sync_metrics),
        stage7_demod_metrics=dict(cand_correct.stage7_demod_metrics),
        stage8_interleaver_metrics=dict(cand_correct.stage8_interleaver_metrics),
        stage9_fec_metrics={"decoder_success": False, "fec_quality_score": 0.10, "normalized_path_metric": 0.50},
        stage10_validation_metrics={
            "crc_available": 1.0,
            "crc_pass": 0.0,
            "crc_pass_rate": 0.0,
            "syndrome_valid": 0.0,
            "contradiction_count": 2.0,
            "overall_validation_score": 0.10,
            "validation_status": "VALIDATION_FAILED",
        },
    )
    cand_wrong_fec.candidate_correct = evaluate_candidate_correctness(cand_wrong_fec, ground_truth)
    candidates.append(cand_wrong_fec)

    # 4. Competing Negatives (wrong mod / wrong baud)
    mod_options = ["BPSK", "QPSK", "8PSK", "16QAM", "2FSK"]
    rate_options = [4800.0, 9600.0, 19200.0]

    for i in range(num_competing_candidates - 3):
        m = mod_options[(i + 1) % len(mod_options)]
        r = rate_options[(i + 1) % len(rate_options)]
        cand_neg = PipelinePathCandidate(
            signal_id=signal_id,
            candidate_path_id=f"{signal_id}_path_neg_{i}",
            candidate_id=f"cand_neg_{i}",
            modulation=m,
            symbol_rate_hz=r,
            interleaver_family="none",
            fec_family="none",
            stage5_metrics={
                "modulation": m,
                "modulation_probability": float(np.random.uniform(0.05, 0.40)),
                "symbol_rate_score": float(np.random.uniform(0.10, 0.50)),
                "candidate_initial_score": float(np.random.uniform(0.10, 0.40)),
                "rf_family_support": 0.30,
                "constellation_support": 0.30,
                "modulation_rank": float(i + 2),
                "baud_rank": float(i + 2),
            },
            stage6_sync_metrics={
                "timing_lock_score": float(np.random.uniform(0.10, 0.60)),
                "carrier_lock_score": float(np.random.uniform(0.10, 0.50)),
                "overall_sync_score": float(np.random.uniform(0.10, 0.55)),
                "constellation_quality_after": float(np.random.uniform(0.10, 0.50)),
            },
            stage7_demod_metrics={
                "evm_percent": float(np.random.uniform(25.0, 45.0)),
                "evm_db": -8.0,
                "mean_abs_llr": 0.8,
                "demod_quality_score": 0.25,
            },
            stage8_interleaver_metrics={"interleaver_structural_score": 0.15},
            stage9_fec_metrics={"decoder_success": False, "fec_quality_score": 0.05},
            stage10_validation_metrics={
                "crc_available": 1.0,
                "crc_pass": 0.0,
                "crc_pass_rate": 0.0,
                "syndrome_valid": 0.0,
                "contradiction_count": 3.0,
                "overall_validation_score": float(np.random.uniform(0.0, 0.20)),
                "validation_status": "VALIDATION_FAILED",
            },
        )
        cand_neg.candidate_correct = evaluate_candidate_correctness(cand_neg, ground_truth)
        candidates.append(cand_neg)

    return candidates, ground_truth
