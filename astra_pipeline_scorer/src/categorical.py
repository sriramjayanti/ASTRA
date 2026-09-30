"""
categorical.py
Categorical encoders and mappings for ASTRA Stage 11 — Pipeline Scoring Model.
Provides explicit, versioned numerical encodings for categorical parameters across stages.
"""

from typing import Dict, Any

MODULATION_ENCODING: Dict[str, float] = {
    "BPSK": 0.0,
    "QPSK": 1.0,
    "8PSK": 2.0,
    "16QAM": 3.0,
    "64QAM": 4.0,
    "2FSK": 5.0,
    "4FSK": 6.0,
    "UNKNOWN": 7.0,
}

MODULATION_FAMILY_ENCODING: Dict[str, float] = {
    "PSK": 0.0,
    "QAM": 1.0,
    "FSK": 2.0,
    "OTHER": 3.0,
}

INTERLEAVER_FAMILY_ENCODING: Dict[str, float] = {
    "none": 0.0,
    "identity": 0.0,
    "block": 1.0,
    "convolutional": 2.0,
    "helical": 3.0,
    "diagonal": 3.0,
    "pseudo_random": 4.0,
    "unknown": 5.0,
}

FEC_FAMILY_ENCODING: Dict[str, float] = {
    "none": 0.0,
    "uncoded": 0.0,
    "convolutional": 1.0,
    "reed_solomon": 2.0,
    "concatenated": 3.0,
    "ldpc": 4.0,
    "turbo": 5.0,
    "polar": 6.0,
    "unknown": 7.0,
}

VALIDATION_STATUS_ENCODING: Dict[str, float] = {
    "VALIDATION_STRONG": 4.0,
    "VALIDATION_MODERATE": 3.0,
    "VALIDATION_WEAK": 2.0,
    "VALIDATION_INCONCLUSIVE": 1.0,
    "VALIDATION_FAILED": 0.0,
}


def encode_modulation(mod_name: str) -> float:
    cleaned = str(mod_name).upper().strip()
    return MODULATION_ENCODING.get(cleaned, MODULATION_ENCODING["UNKNOWN"])


def encode_interleaver_family(family: str) -> float:
    cleaned = str(family).lower().strip()
    return INTERLEAVER_FAMILY_ENCODING.get(cleaned, INTERLEAVER_FAMILY_ENCODING["unknown"])


def encode_fec_family(family: str) -> float:
    cleaned = str(family).lower().strip()
    return FEC_FAMILY_ENCODING.get(cleaned, FEC_FAMILY_ENCODING["unknown"])


def encode_validation_status(status: str) -> float:
    cleaned = str(status).upper().strip()
    return VALIDATION_STATUS_ENCODING.get(cleaned, 1.0)
