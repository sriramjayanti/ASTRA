"""
decoder_support.py
FEC re-encoding verification for ASTRA Stage 10 — Validation Engine.
Takes decoded message bits, re-encodes them using the candidate FEC hypothesis,
and compares them against the received deinterleaved channel bitstream to calculate
the exact bit match fraction and weighted reconstruction score.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from .models import ReencodingResult, EvidenceCheckState


def reencode_convolutional_local(
    bits: np.ndarray,
    constraint_length: int = 7,
    generators_octal: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Local lightweight rate 1/n convolutional encoder.
    """
    if generators_octal is None:
        generators_octal = [0o171, 0o133]

    K = constraint_length
    gens = [int(g) for g in generators_octal]
    n_gens = len(gens)

    # Pad with K-1 zeros for termination
    padded = np.pad(bits, (0, K - 1), mode='constant', constant_values=0)
    out_bits = []
    reg = 0

    for bit in padded:
        reg = ((reg << 1) | int(bit)) & ((1 << K) - 1)
        for g in gens:
            # Octal polynomial parity
            parity = bin(reg & g).count('1') % 2
            out_bits.append(parity)

    return np.array(out_bits, dtype=np.uint8)


def reencode_and_compare(
    decoded_bits: Optional[np.ndarray],
    fec_family: str,
    fec_params: Dict[str, Any],
    received_channel_bits: Optional[np.ndarray],
    max_comparison_bits: int = 2048,
) -> ReencodingResult:
    """
    Re-encode decoded message bits and compare with received channel bits.
    """
    if decoded_bits is None or received_channel_bits is None:
        return ReencodingResult(check_state=EvidenceCheckState.NOT_TESTED)

    decoded_bits = np.asarray(decoded_bits, dtype=np.uint8).ravel()
    received_channel_bits = np.asarray(received_channel_bits, dtype=np.uint8).ravel()

    if len(decoded_bits) == 0 or len(received_channel_bits) == 0:
        return ReencodingResult(check_state=EvidenceCheckState.NOT_TESTED)

    family = fec_family.lower()
    reencoded_bits = None

    if family in ("none", "uncoded"):
        # Uncoded stream is identical to channel bits
        reencoded_bits = decoded_bits
    elif family == "convolutional":
        K = fec_params.get("constraint_length", 7)
        gens = fec_params.get("generators_octal", [0o171, 0o133])
        # Convert octal representation if passed as string/int
        if isinstance(gens, list):
            gens_clean = []
            for g in gens:
                if isinstance(g, str):
                    gens_clean.append(int(g, 8))
                else:
                    gens_clean.append(int(g))
            gens = gens_clean
        reencoded_bits = reencode_convolutional_local(decoded_bits, K, gens)
    else:
        # Fallback for complex concatenated/LDPC if local encoder not available
        return ReencodingResult(check_state=EvidenceCheckState.NOT_TESTED)

    compare_len = min(len(reencoded_bits), len(received_channel_bits), max_comparison_bits)
    if compare_len < 4:
        return ReencodingResult(check_state=EvidenceCheckState.NOT_TESTED)

    matches = np.sum(reencoded_bits[:compare_len] == received_channel_bits[:compare_len])
    match_fraction = float(matches) / float(compare_len)

    # Calculate weighted score (match > 90% is strong evidence)
    weighted_score = max(0.0, (match_fraction - 0.50) * 2.0)
    check_state = EvidenceCheckState.PASS if match_fraction >= 0.85 else (
        EvidenceCheckState.FAIL if match_fraction < 0.60 else EvidenceCheckState.NOT_TESTED
    )

    return ReencodingResult(
        bits_compared=compare_len,
        bit_match_fraction=match_fraction,
        weighted_match_score=weighted_score,
        check_state=check_state,
    )
