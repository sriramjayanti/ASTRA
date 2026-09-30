"""
ASTRA Label Mapping for Broad Signal Families and Quality Ground Truth.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


FAMILY_MAPPING: Dict[str, str] = {
    # FSK Family
    "2-FSK": "FSK",
    "4-FSK": "FSK",
    "8-FSK": "FSK",
    "2FSK": "FSK",
    "4FSK": "FSK",
    "8FSK": "FSK",
    "GFSK": "FSK",
    "MSK": "FSK",

    # PSK Family
    "BPSK": "PSK",
    "QPSK": "PSK",
    "8PSK": "PSK",
    "16PSK": "PSK",
    "OQPSK": "PSK",
    "PI/4-QPSK": "PSK",
    "PI4QPSK": "PSK",
    "DQPSK": "PSK",
    "2PSK": "PSK",
    "4PSK": "PSK",

    # QAM Family
    "16-QAM": "QAM",
    "64-QAM": "QAM",
    "256-QAM": "QAM",
    "16QAM": "QAM",
    "64QAM": "QAM",
    "256QAM": "QAM",
    "4QAM": "PSK",  # 4-QAM is geometrically equivalent to QPSK

    # Unknown / Noise / CW
    "NOISE": "UNKNOWN",
    "CW": "UNKNOWN",
    "UNKNOWN": "UNKNOWN",
    "NONE": "UNKNOWN",
}


def map_modulation_to_family(modulation_name: str) -> str:
    """
    Maps an exact modulation name (e.g. '16-QAM' or '4-FSK') to broad family ('QAM', 'FSK', 'PSK', 'UNKNOWN').
    """
    if not modulation_name:
        return "UNKNOWN"
    norm = str(modulation_name).strip().upper()
    return FAMILY_MAPPING.get(norm, "UNKNOWN")


def extract_quality_labels(
    meta: Dict[str, Any],
    low_snr_threshold_db: float = 3.0,
    clipping_threshold: float = 0.05,
    cfo_affected_threshold_hz: float = 500.0,
) -> Dict[str, int]:
    """
    Extracts binary multi-label quality ground truth from signal metadata.
    """
    snr_db = float(meta.get("snr_db", meta.get("true_snr_db", 20.0)))
    cfo_hz = float(abs(meta.get("cfo_hz", meta.get("true_cfo_hz", 0.0))))
    clipping_ratio = float(meta.get("clipping_ratio", meta.get("clip_ratio", 0.0)))
    is_multipath = bool(meta.get("multipath", meta.get("is_multipath", False)))

    return {
        "low_snr": 1 if snr_db <= low_snr_threshold_db else 0,
        "clipped": 1 if clipping_ratio >= clipping_threshold else 0,
        "multipath": 1 if is_multipath else 0,
        "cfo_affected": 1 if cfo_hz >= cfo_affected_threshold_hz else 0,
    }
