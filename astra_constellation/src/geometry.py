"""
ASTRA Constellation Geometry, Symmetry, and EVM Calculators.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


def get_reference_constellation(constellation_type: str) -> np.ndarray:
    """
    Returns unit-average-power reference constellation points.
    """
    ctype = constellation_type.upper().replace("-", "")

    if ctype in ["BPSK", "2PSK"]:
        pts = np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
    elif ctype in ["QPSK", "4QAM", "4PSK"]:
        grid = np.array([-1.0, 1.0])
        i_grid, q_grid = np.meshgrid(grid, grid)
        pts = (i_grid.flatten() + 1j * q_grid.flatten()) / np.sqrt(2.0)
    elif ctype in ["8PSK"]:
        angles = np.arange(8) * (2 * np.pi / 8.0)
        pts = np.cos(angles) + 1j * np.sin(angles)
    elif ctype in ["16QAM"]:
        grid = np.array([-3.0, -1.0, 1.0, 3.0])
        i_grid, q_grid = np.meshgrid(grid, grid)
        pts = (i_grid.flatten() + 1j * q_grid.flatten()) / np.sqrt(10.0)
    elif ctype in ["64QAM"]:
        grid = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
        i_grid, q_grid = np.meshgrid(grid, grid)
        pts = (i_grid.flatten() + 1j * q_grid.flatten()) / np.sqrt(42.0)
    else:
        # Default QPSK fallback
        grid = np.array([-1.0, 1.0])
        i_grid, q_grid = np.meshgrid(grid, grid)
        pts = (i_grid.flatten() + 1j * q_grid.flatten()) / np.sqrt(2.0)

    return pts.astype(np.complex64)


def compute_evm(iq: np.ndarray, reference_constellation: np.ndarray) -> float:
    """
    Calculates Error Vector Magnitude (EVM) in percentage.
    EVM = sqrt( mean( |x - nearest_ref|^2 ) / P_ref ) * 100%
    """
    if len(iq) == 0 or len(reference_constellation) == 0:
        return 100.0

    # Ensure signal is normalized to unit average power
    pwr = np.mean(np.abs(iq) ** 2)
    if pwr < 1e-12:
        return 100.0
    x_norm = iq / np.sqrt(pwr)

    # Fast nearest neighbor distance
    # shape [N, 1] - shape [1, M] -> [N, M]
    diffs = np.abs(x_norm[:, None] - reference_constellation[None, :]) ** 2
    min_dist_sq = np.min(diffs, axis=1)

    evm_rms = np.sqrt(np.mean(min_dist_sq))
    return float(np.clip(evm_rms * 100.0, 0.0, 100.0))


def compute_angular_uniformity(iq: np.ndarray, m_ary: int = 4) -> float:
    """
    Measures phase concentration at m_ary angular sectors.
    Returns uniformity score in [0.0, 1.0] (1.0 = sharp phase alignment).
    """
    phases = np.angle(iq)  # [-pi, +pi]
    sector = 2.0 * np.pi / max(1, m_ary)
    
    # Modulo sector phase dispersion
    mod_phases = (phases + np.pi) % sector
    phase_var = float(np.var(mod_phases))
    max_var = (sector ** 2) / 12.0  # Variance of uniform distribution over sector

    uniformity = float(np.clip(1.0 - (phase_var / (max_var + 1e-12)), 0.0, 1.0))
    return uniformity


def compute_square_grid_compactness(iq: np.ndarray) -> float:
    """
    Measures Cartesian grid lattice structure for QAM constellations.
    """
    x = np.real(iq)
    y = np.imag(iq)
    if len(x) < 32:
        return 0.0

    # Kurtosis of I and Q (for uniform lattice, kurtosis is lower than Gaussian)
    kurt_i = float(np.mean((x - np.mean(x)) ** 4) / (np.var(x) ** 2 + 1e-12))
    kurt_q = float(np.mean((y - np.mean(y)) ** 4) / (np.var(y) ** 2 + 1e-12))
    
    # For Gaussian noise kurtosis ~ 3.0; for discrete uniform lattice kurtosis < 2.2
    score_i = float(np.clip((3.0 - kurt_i) / 1.5, 0.0, 1.0))
    score_q = float(np.clip((3.0 - kurt_q) / 1.5, 0.0, 1.0))
    
    return float((score_i + score_q) / 2.0)
