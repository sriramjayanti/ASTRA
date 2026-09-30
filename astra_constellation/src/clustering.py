"""
ASTRA Constellation Spatial Clustering and Radial Ring Analyzers.
Implements K-Means, DBSCAN, and Amplitude Ring Analysis on the IQ Plane.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


def extract_iq_coordinates(iq: np.ndarray, max_samples: int = 4096) -> np.ndarray:
    """
    Converts 1D complex IQ array into an [N, 2] real coordinate array (I, Q).
    """
    x = np.asarray(iq)
    if len(x) > max_samples:
        step = len(x) // max_samples
        x = x[::step][:max_samples]
    
    # Normalize to unit mean power for consistent spatial analysis
    pwr = np.mean(np.abs(x) ** 2)
    if pwr > 1e-12:
        x = x / np.sqrt(pwr)

    return np.column_stack([np.real(x), np.imag(x)]).astype(np.float32)


def evaluate_kmeans_clustering(
    coords: np.ndarray,
    k_values: Sequence[int] = (2, 4, 8, 16, 64),
    random_state: int = 42,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Evaluates K-Means clustering across candidate constellation orders M.
    Returns:
        Tuple[silhouette_scores_dict, inertias_dict]
    """
    if len(coords) < 64:
        return {k: 0.0 for k in k_values}, {k: 0.0 for k in k_values}

    silhouette_scores: Dict[int, float] = {}
    inertias: Dict[int, float] = {}

    for k in k_values:
        if len(coords) <= k:
            silhouette_scores[k] = 0.0
            inertias[k] = 0.0
            continue

        try:
            km = KMeans(n_clusters=k, n_init=3, max_iter=50, random_state=random_state)
            labels = km.fit_predict(coords)
            inertias[k] = float(km.inertia_)

            # Compute Silhouette score on a fast subset if large
            if len(set(labels)) > 1:
                sub_sample = min(len(coords), 1000)
                score = float(silhouette_score(coords[:sub_sample], labels[:sub_sample]))
                silhouette_scores[k] = max(0.0, score)
            else:
                silhouette_scores[k] = 0.0
        except Exception:
            silhouette_scores[k] = 0.0
            inertias[k] = 0.0

    return silhouette_scores, inertias


def evaluate_dbscan_clustering(
    coords: np.ndarray,
    eps: float = 0.15,
    min_samples: int = 15,
) -> Tuple[int, float]:
    """
    Performs unconstrained density clustering to discover natural spatial cluster count.
    Returns:
        Tuple[cluster_count, noise_ratio]
    """
    if len(coords) < min_samples:
        return 0, 1.0

    try:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(coords)
        unique_labels = set(labels)
        
        # Noise samples are labeled as -1
        is_noise = (labels == -1)
        noise_ratio = float(np.mean(is_noise))
        cluster_count = len([l for l in unique_labels if l != -1])
        return cluster_count, noise_ratio
    except Exception:
        return 0, 1.0


def extract_radial_rings(
    iq: np.ndarray,
    num_bins: int = 64,
    prominence_ratio: float = 0.08,
) -> Tuple[int, List[float], float]:
    """
    Analyzes the amplitude distribution r = |x[n]| to identify concentric energy rings.
    
    Returns:
        Tuple[ring_count, ring_amplitudes_list, constant_modulus_variance]
    """
    amplitudes = np.abs(iq)
    if len(amplitudes) < 32:
        return 0, [], 1.0

    mean_amp = np.mean(amplitudes)
    if mean_amp > 1e-12:
        norm_amp = amplitudes / mean_amp
    else:
        norm_amp = amplitudes

    cm_variance = float(np.var(norm_amp))

    # Histogram of normalized amplitudes
    hist, bin_edges = np.histogram(norm_amp, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    max_hist = np.max(hist) if len(hist) > 0 else 1.0
    prom = max_hist * prominence_ratio

    peaks, props = find_peaks(hist, prominence=prom, distance=3)
    ring_amplitudes = [float(round(bin_centers[p], 3)) for p in peaks]
    ring_count = len(ring_amplitudes)

    return ring_count, ring_amplitudes, cm_variance
