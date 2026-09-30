"""
ASTRA Central Canonical Modulation Class Set & Mappings (V2).
Single source of truth for all modulation classification models, datasets,
evaluation metrics, fusion, and GUI displays.
"""

from __future__ import annotations
from typing import Dict, List, Optional

# 10 Trained Classification Targets in strict, immutable canonical order
TRAINED_MODULATION_CLASSES_V2: List[str] = [
    "2-FSK",
    "4-FSK",
    "BPSK",
    "QPSK",
    "8PSK",
    "DQPSK",
    "MSK",
    "16QAM",
    "64QAM",
    "256QAM",
]

UNKNOWN_CLASS: str = "UNKNOWN"

# Full Common Class Set including UNKNOWN detection
MODULATION_CLASSES_V2: List[str] = TRAINED_MODULATION_CLASSES_V2 + [UNKNOWN_CLASS]

# Fast Lookup Mappings
CLASS_TO_IDX_V2: Dict[str, int] = {cls: idx for idx, cls in enumerate(TRAINED_MODULATION_CLASSES_V2)}
IDX_TO_CLASS_V2: Dict[int, str] = {idx: cls for idx, cls in enumerate(TRAINED_MODULATION_CLASSES_V2)}

# Canonical Alias Mapping for legacy, file-system, or dataset variations
MODULATION_ALIASES: Dict[str, str] = {
    # FSK
    "2fsk": "2-FSK",
    "2-fsk": "2-FSK",
    "fsk2": "2-FSK",
    "fsk-2": "2-FSK",
    "4fsk": "4-FSK",
    "4-fsk": "4-FSK",
    "fsk4": "4-FSK",
    "fsk-4": "4-FSK",
    # PSK
    "bpsk": "BPSK",
    "qpsk": "QPSK",
    "8psk": "8PSK",
    "8-psk": "8PSK",
    "dqpsk": "DQPSK",
    "d-qpsk": "DQPSK",
    "msk": "MSK",
    # QAM
    "16qam": "16QAM",
    "16-qam": "16QAM",
    "qam16": "16QAM",
    "64qam": "64QAM",
    "64-qam": "64QAM",
    "qam64": "64QAM",
    "256qam": "256QAM",
    "256-qam": "256QAM",
    "qam256": "256QAM",
    # Unknown
    "unknown": "UNKNOWN",
    "none": "UNKNOWN",
    "noise": "UNKNOWN",
}


def normalize_modulation_name(name: str) -> str:
    """
    Normalizes arbitrary modulation name strings into the canonical V2 format.
    
    Examples:
        '2fsk' -> '2-FSK'
        '16-qam' -> '16QAM'
        'bpsk' -> 'BPSK'
    """
    if not name:
        return UNKNOWN_CLASS
    clean = name.strip()
    # Check direct match
    if clean in MODULATION_CLASSES_V2:
        return clean
    # Check lowercase alias lookup
    lower = clean.lower().replace("_", "-")
    if lower in MODULATION_ALIASES:
        return MODULATION_ALIASES[lower]
    lower_plain = lower.replace("-", "")
    if lower_plain in MODULATION_ALIASES:
        return MODULATION_ALIASES[lower_plain]
    return clean.upper()


def get_class_index(name: str) -> Optional[int]:
    """Returns the zero-based index for a trained modulation class or None if unknown."""
    norm = normalize_modulation_name(name)
    return CLASS_TO_IDX_V2.get(norm)
