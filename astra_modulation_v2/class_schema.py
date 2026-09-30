"""
Authoritative Class Schema for ASTRA Modulation Intelligence V2.

Canonical Class Set (Exactly 11 Classes):
1.  2-FSK
2.  4-FSK
3.  BPSK
4.  QPSK
5.  8PSK
6.  DQPSK
7.  MSK
8.  16QAM
9.  64QAM
10. 256QAM
11. UNKNOWN

This file is the single source of truth across dataset generation, training,
checkpoint metadata, inference, fusion, candidate engine, GUI, and explainability.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

CLASS_SCHEMA_VERSION: str = "modulation_classes_v2"
NUM_CLASSES_V2: int = 11

MODULATION_CLASSES_V2: List[str] = [
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
    "UNKNOWN",
]

CLASS_TO_IDX_V2: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(MODULATION_CLASSES_V2)}
IDX_TO_CLASS_V2: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(MODULATION_CLASSES_V2)}

# Modulation Family Groupings for RF Support & Hierarchical Validation
MODULATION_FAMILIES_V2: Dict[str, str] = {
    "2-FSK": "FSK",
    "4-FSK": "FSK",
    "MSK": "FSK",  # Continuous-phase frequency shift
    "BPSK": "PSK",
    "QPSK": "PSK",
    "8PSK": "PSK",
    "DQPSK": "PSK",
    "16QAM": "QAM",
    "64QAM": "QAM",
    "256QAM": "QAM",
    "UNKNOWN": "NON_TARGET",
}

# Alias mapping for robust normalization of dataset, CSPB, and legacy names
_CLASS_ALIASES: Dict[str, str] = {
    "2-fsk": "2-FSK",
    "2fsk": "2-FSK",
    "fsk2": "2-FSK",
    "4-fsk": "4-FSK",
    "4fsk": "4-FSK",
    "fsk4": "4-FSK",
    "bpsk": "BPSK",
    "qpsk": "QPSK",
    "8psk": "8PSK",
    "8-psk": "8PSK",
    "dqpsk": "DQPSK",
    "d-qpsk": "DQPSK",
    "msk": "MSK",
    "16qam": "16QAM",
    "16-qam": "16QAM",
    "64qam": "64QAM",
    "64-qam": "64QAM",
    "256qam": "256QAM",
    "256-qam": "256QAM",
    "unknown": "UNKNOWN",
    "noise": "UNKNOWN",
    "awgn": "UNKNOWN",
    "cw": "UNKNOWN",
    "chirp": "UNKNOWN",
    "multitone": "UNKNOWN",
    "multi-tone": "UNKNOWN",
    "non_target": "UNKNOWN",
    "interference": "UNKNOWN",
}


def normalize_modulation_name(name: str) -> str:
    """Normalizes any incoming modulation string to the canonical V2 class name."""
    cleaned = str(name).strip().lower().replace("_", "-")
    cleaned_no_dash = str(name).strip().lower().replace("_", "").replace("-", "")
    
    if cleaned in _CLASS_ALIASES:
        return _CLASS_ALIASES[cleaned]
    if cleaned_no_dash in _CLASS_ALIASES:
        return _CLASS_ALIASES[cleaned_no_dash]
    
    # Direct case-insensitive match against canonical classes
    for c in MODULATION_CLASSES_V2:
        if c.lower() == cleaned or c.lower().replace("-", "") == cleaned_no_dash:
            return c
            
    # Default fallback for unhandled or anomalous non-target signals
    return "UNKNOWN"


def get_class_index(name: str) -> int:
    """Returns integer label [0..10] for any modulation string."""
    canonical = normalize_modulation_name(name)
    return CLASS_TO_IDX_V2[canonical]


def get_class_name(idx: int) -> str:
    """Returns canonical class string for given index [0..10]."""
    if idx not in IDX_TO_CLASS_V2:
        raise ValueError(f"Invalid class index {idx}. Valid range is 0 to {NUM_CLASSES_V2-1}.")
    return IDX_TO_CLASS_V2[idx]


def assert_runtime_class_order(classes: List[str]) -> None:
    """Asserts that provided class list strictly matches MODULATION_CLASSES_V2."""
    if len(classes) != len(MODULATION_CLASSES_V2):
        raise ValueError(
            f"Class count mismatch: expected {len(MODULATION_CLASSES_V2)}, got {len(classes)}. "
            f"Expected {MODULATION_CLASSES_V2}, got {classes}"
        )
    for i, (c_act, c_exp) in enumerate(zip(classes, MODULATION_CLASSES_V2)):
        if c_act != c_exp:
            raise ValueError(
                f"Class order mismatch at index {i}: expected '{c_exp}', got '{c_act}'. "
                f"Canonical order must be: {MODULATION_CLASSES_V2}"
            )
