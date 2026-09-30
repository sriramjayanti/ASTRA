"""
Multipath FIR Channel Impairment Module for ASTRA Engine 6.
Constructs discrete tapped-delay-line channel models with energy normalization.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


MULTIPATH_PROFILES: dict[str, dict[str, Any]] = {
    "mild": {
        "delays_samples": [0, 2, 5],
        "amplitudes": [1.0 + 0j, 0.3 * np.exp(1j * 0.5), 0.15 * np.exp(-1j * 0.8)],
    },
    "medium": {
        "delays_samples": [0, 3, 7, 12],
        "amplitudes": [
            1.0 + 0j,
            0.5 * np.exp(1j * 1.2),
            0.3 * np.exp(-1j * 0.4),
            0.1 * np.exp(1j * 2.0),
        ],
    },
    "severe": {
        "delays_samples": [0, 1, 4, 8, 16],
        "amplitudes": [
            1.0 + 0j,
            0.7 * np.exp(1j * 2.1),
            0.5 * np.exp(-1j * 1.5),
            0.3 * np.exp(1j * 0.9),
            0.2 * np.exp(-1j * 2.5),
        ],
    },
}


def create_multipath_impulse_response(
    delays_samples: Sequence[int],
    amplitudes: Sequence[complex],
    normalize_energy: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a discrete-time FIR channel impulse response h[n] from tap delays and complex amplitudes.

    Args:
        delays_samples: Integer sample delay offsets for each tap (first tap typically 0).
        amplitudes: Complex gain coefficients for each tap.
        normalize_energy: If True, scale taps so that sum(|h_k|^2) = 1.0.

    Returns:
        tuple (cir_h, normalized_taps, delays_array)
    """
    if len(delays_samples) != len(amplitudes):
        raise ValueError("delays_samples and amplitudes must have identical length")
    if len(delays_samples) == 0:
        return np.array([1.0 + 0j], dtype=np.complex64), np.array([1.0 + 0j], dtype=np.complex64), np.array([0], dtype=np.int64)

    max_delay = int(max(delays_samples))
    cir = np.zeros(max_delay + 1, dtype=np.complex128)
    taps_arr = np.array(amplitudes, dtype=np.complex128)
    delays_arr = np.array(delays_samples, dtype=np.int64)

    for d, a in zip(delays_samples, amplitudes):
        if d < 0:
            raise ValueError(f"Tap delay must be non-negative, got {d}")
        cir[int(d)] += complex(a)

    pre_norm_energy = float(np.sum(np.abs(cir) ** 2))

    if normalize_energy and pre_norm_energy > 1e-12:
        norm_factor = np.sqrt(pre_norm_energy)
        cir = cir / norm_factor
        taps_arr = taps_arr / norm_factor

    return cir.astype(np.complex64), taps_arr.astype(np.complex64), delays_arr


def apply_multipath(
    iq: np.ndarray,
    delays_samples: Sequence[int] | None = None,
    amplitudes: Sequence[complex] | None = None,
    taps: Sequence[complex] | None = None,
    profile: str | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply multipath FIR convolution to baseband IQ.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        delays_samples: Optional explicit delay list.
        amplitudes: Optional explicit complex tap list (alias for taps).
        taps: Optional explicit complex tap list.
        profile: Optional predefined profile name ('mild', 'medium', 'severe').
        normalize: Whether to normalize channel impulse response energy.

    Returns:
        tuple (multipath_iq, channel_taps, delay_offsets)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), np.array([1.0 + 0j], dtype=np.complex64), np.array([0], dtype=np.int64)

    if profile is not None and profile.lower() in MULTIPATH_PROFILES:
        prof = MULTIPATH_PROFILES[profile.lower()]
        delays = prof["delays_samples"]
        amps = prof["amplitudes"]
    elif delays_samples is not None and (taps is not None or amplitudes is not None):
        delays = delays_samples
        amps = taps if taps is not None else amplitudes
    else:
        # Default mild multipath
        prof = MULTIPATH_PROFILES["mild"]
        delays = prof["delays_samples"]
        amps = prof["amplitudes"]

    cir, norm_taps, delays_out = create_multipath_impulse_response(
        delays_samples=delays, amplitudes=amps, normalize_energy=normalize
    )

    convolved = np.convolve(iq, cir, mode="full")
    # Output length preserved to match input length
    multipath_iq = convolved[: len(iq)].astype(np.complex64)

    return multipath_iq, norm_taps, delays_out
