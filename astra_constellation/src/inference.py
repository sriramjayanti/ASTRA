"""
ASTRA Constellation Analysis Engine Unified Inference API.
Combines K-Means, DBSCAN, Radial Ring, and EVM metrics to provide explainable modulation support.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import yaml

from .clustering import (
    evaluate_dbscan_clustering,
    evaluate_kmeans_clustering,
    extract_iq_coordinates,
    extract_radial_rings,
)
from .geometry import (
    compute_angular_uniformity,
    compute_evm,
    compute_square_grid_compactness,
    get_reference_constellation,
)
from .models import (
    ConstellationCandidate,
    ConstellationEvidence,
    ConstellationPrediction,
    InvalidIQError,
)

logger = logging.getLogger("astra_constellation.inference")


class ConstellationAnalyzer:
    """
    Analyzes spatial IQ scatter to extract cluster order, energy rings, and EVM.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config: Dict[str, Any] = {}
        self.model_version = "astra_constellation_v1.0"

        if config_path is not None and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            default_cfg = Path(__file__).resolve().parent.parent / "configs" / "constellation_config.yaml"
            if default_cfg.exists():
                with open(default_cfg, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}

    def analyze(self, iq: np.ndarray) -> ConstellationPrediction:
        """
        Extracts spatial clustering metrics and ranks constellation hypotheses.
        """
        if iq is None or len(iq) == 0:
            raise InvalidIQError("Input IQ array is empty or None.")
        if not np.all(np.isfinite(iq)):
            raise InvalidIQError("Input IQ array contains NaN or infinite values.")

        x = np.asarray(iq)
        if len(x) < 32:
            return ConstellationPrediction(
                best_constellation="UNKNOWN",
                m_ary_order=0,
                confidence=0.0,
                status="UNKNOWN",
                top_k=[],
                evidence={},
                unknown_reason="Signal length too short for spatial clustering.",
            )

        coords = extract_iq_coordinates(x, max_samples=4096)

        # 1. K-Means Multi-K Evaluation
        k_values = self.config.get("constellation", {}).get("evaluated_k_values", [2, 4, 8, 16, 64])
        sil_scores, inertias = evaluate_kmeans_clustering(coords, k_values=k_values)

        # 2. DBSCAN Density Clustering
        db_cfg = self.config.get("constellation", {}).get("dbscan", {})
        eps = float(db_cfg.get("eps", 0.15))
        min_samples = int(db_cfg.get("min_samples", 15))
        db_clusters, noise_ratio = evaluate_dbscan_clustering(coords, eps=eps, min_samples=min_samples)

        # 3. Radial Ring Analysis
        ring_count, ring_amps, cm_var = extract_radial_rings(x)

        # 4. Geometry & Compactness
        ang_unif_4 = compute_angular_uniformity(x, m_ary=4)
        ang_unif_8 = compute_angular_uniformity(x, m_ary=8)
        grid_compact = compute_square_grid_compactness(x)

        # 5. EVM Calculations against reference templates
        ref_bpsk = get_reference_constellation("BPSK")
        ref_qpsk = get_reference_constellation("QPSK")
        ref_8psk = get_reference_constellation("8PSK")
        ref_16qam = get_reference_constellation("16QAM")
        ref_64qam = get_reference_constellation("64QAM")

        evm_bpsk = compute_evm(x, ref_bpsk)
        evm_qpsk = compute_evm(x, ref_qpsk)
        evm_8psk = compute_evm(x, ref_8psk)
        evm_16qam = compute_evm(x, ref_16qam)
        evm_64qam = compute_evm(x, ref_64qam)

        # Build Evidence Record
        evidence = ConstellationEvidence(
            dbscan_cluster_count=db_clusters,
            dbscan_noise_ratio=noise_ratio,
            kmeans_silhouette_scores=sil_scores,
            kmeans_inertias=inertias,
            radial_ring_count=ring_count,
            radial_ring_amplitudes=ring_amps,
            constant_modulus_variance=cm_var,
            angular_phase_uniformity=ang_unif_4,
            square_grid_compactness=grid_compact,
            estimated_evm_percent=min(evm_bpsk, evm_qpsk, evm_8psk, evm_16qam, evm_64qam),
            sample_count=len(x),
        )

        # 6. Score Constellation Hypotheses
        candidates: List[ConstellationCandidate] = []

        # Candidate: BPSK (M=2)
        score_bpsk = float(
            0.40 * sil_scores.get(2, 0.0) +
            0.30 * (1.0 - min(1.0, evm_bpsk / 50.0)) +
            0.20 * (1.0 if ring_count == 1 else 0.3) +
            0.10 * (1.0 if db_clusters == 2 else 0.4)
        )
        candidates.append(ConstellationCandidate("BPSK", 2, score_bpsk, ["k_means_k2", "evm_bpsk"], evm_bpsk))

        # Candidate: QPSK (M=4)
        score_qpsk = float(
            0.40 * sil_scores.get(4, 0.0) +
            0.30 * (1.0 - min(1.0, evm_qpsk / 50.0)) +
            0.20 * (1.0 if ring_count == 1 else 0.3) +
            0.10 * ang_unif_4
        )
        candidates.append(ConstellationCandidate("QPSK", 4, score_qpsk, ["k_means_k4", "evm_qpsk"], evm_qpsk))

        # Candidate: 8PSK (M=8)
        score_8psk = float(
            0.40 * sil_scores.get(8, 0.0) +
            0.30 * (1.0 - min(1.0, evm_8psk / 50.0)) +
            0.20 * (1.0 if ring_count == 1 else 0.3) +
            0.10 * ang_unif_8
        )
        candidates.append(ConstellationCandidate("8PSK", 8, score_8psk, ["k_means_k8", "evm_8psk"], evm_8psk))

        # Candidate: 16-QAM (M=16)
        score_16qam = float(
            0.35 * sil_scores.get(16, 0.0) +
            0.30 * (1.0 - min(1.0, evm_16qam / 50.0)) +
            0.20 * (1.0 if ring_count == 3 else 0.4) +
            0.15 * grid_compact
        )
        candidates.append(ConstellationCandidate("16-QAM", 16, score_16qam, ["k_means_k16", "radial_3_rings", "grid_lattice"], evm_16qam))

        # Candidate: 64-QAM (M=64)
        score_64qam = float(
            0.35 * sil_scores.get(64, 0.0) +
            0.30 * (1.0 - min(1.0, evm_64qam / 50.0)) +
            0.20 * (1.0 if ring_count >= 6 else 0.3) +
            0.15 * grid_compact
        )
        candidates.append(ConstellationCandidate("64-QAM", 64, score_64qam, ["k_means_k64", "grid_lattice"], evm_64qam))

        # Sort descending by score
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]

        # Determine ASTRA Status & Confidence
        top1 = best.score
        top2 = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top1 - top2

        if top1 >= 0.70 and margin >= 0.12:
            status = "CONFIRMED"
        elif top1 >= 0.45:
            status = "ESTIMATED"
        elif top1 >= 0.25:
            status = "POSSIBLE"
        else:
            status = "UNKNOWN"

        top_k_list = [c.to_dict() for c in candidates[:3]]

        return ConstellationPrediction(
            best_constellation=best.constellation_type,
            m_ary_order=best.m_ary_order,
            confidence=float(round(top1, 4)),
            status=status,
            top_k=top_k_list,
            evidence=evidence.to_dict(),
            model_version=self.model_version,
            unknown_reason=None if status != "UNKNOWN" else "Ambiguous spatial distribution",
        )
