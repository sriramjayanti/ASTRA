"""
Flat Rayleigh and Rician Fading Channel Modules for ASTRA Engine 6.
Implements unit-expected-power statistical fading with line-of-sight and scattered components.
"""

from __future__ import annotations

import numpy as np


def apply_rayleigh_fading(
    iq: np.ndarray,
    seed: int | None = None,
) -> tuple[np.ndarray, complex]:
    """Apply flat Rayleigh fading to baseband IQ.

    Fading channel coefficient h ~ CN(0, 1) such that E[|h|^2] = 1.0.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        seed: Random seed for reproducibility.

    Returns:
        tuple (faded_iq, fading_coefficient_h)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), 1.0 + 0.0j

    rng = np.random.default_rng(seed)
    # Generate CN(0, 1): h = (X + j*Y) / sqrt(2) where X, Y ~ N(0, 1)
    x = rng.normal(loc=0.0, scale=1.0 / np.sqrt(2.0))
    y = rng.normal(loc=0.0, scale=1.0 / np.sqrt(2.0))
    h = complex(x, y)

    faded_iq = (iq * h).astype(np.complex64)
    return faded_iq, h


def apply_rician_fading(
    iq: np.ndarray,
    k_factor_db: float = 8.0,
    los_phase_rad: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, complex, float]:
    """Apply flat Rician fading with Line-of-Sight (LOS) dominance.

    Rician factor K = P_los / P_diffuse
    Total coefficient h = sqrt(K / (K+1)) * exp(j*phi_los) + sqrt(1 / (K+1)) * CN(0, 1)
    Ensures E[|h|^2] = 1.0.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        k_factor_db: Rician K-factor in dB (e.g. 0 dB, 5 dB, 10 dB, 20 dB).
        los_phase_rad: Line-of-sight component phase angle in radians.
        seed: Random seed for reproducibility.

    Returns:
        tuple (faded_iq, fading_coefficient_h, k_factor_db)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), 1.0 + 0.0j, float(k_factor_db)

    rng = np.random.default_rng(seed)
    k_linear = 10.0 ** (k_factor_db / 10.0)

    los_weight = np.sqrt(k_linear / (k_linear + 1.0))
    diffuse_weight = np.sqrt(1.0 / (k_linear + 1.0))

    # Line of sight component
    h_los = los_weight * np.exp(1j * los_phase_rad)

    # Diffuse scattered component ~ CN(0, 1)
    x = rng.normal(loc=0.0, scale=1.0 / np.sqrt(2.0))
    y = rng.normal(loc=0.0, scale=1.0 / np.sqrt(2.0))
    h_diffuse = diffuse_weight * complex(x, y)

    h = complex(h_los + h_diffuse)
    faded_iq = (iq * h).astype(np.complex64)

    return faded_iq, h, float(k_factor_db)
