"""
syndrome.py
FEC syndrome and decoder telemetry integration for ASTRA Stage 10.
Consumes telemetry produced in Stage 9 (RS syndromes, LDPC H*c=0, Viterbi path metrics)
and normalizes it into standardized evidence without repeating full decoding.
"""

from typing import Any, Dict, Optional
import numpy as np
from .models import SyndromeResult, EvidenceCheckState


def normalize_syndrome_evidence(fec_candidate: Any) -> SyndromeResult:
    """
    Extract and normalize FEC syndrome/decoder telemetry from a Stage 9 FECCandidateResult.
    """
    if fec_candidate is None:
        return SyndromeResult(check_state=EvidenceCheckState.NOT_TESTED)

    # Extract metrics safely whether dict or object
    if hasattr(fec_candidate, "fec_family"):
        family = getattr(fec_candidate, "fec_family", "none")
        decoder_success = getattr(fec_candidate, "decoder_success", False)
        syndrome_weight = getattr(fec_candidate, "syndrome_weight", 0.0)
        parity_check_success = getattr(fec_candidate, "parity_check_success", False)
        path_metric = getattr(fec_candidate, "path_metric", 0.0)
        decoder_metrics = getattr(fec_candidate, "decoder_metrics", {}) or {}
    elif isinstance(fec_candidate, dict):
        family = fec_candidate.get("fec_family", "none")
        decoder_success = fec_candidate.get("decoder_success", False)
        syndrome_weight = fec_candidate.get("syndrome_weight", 0.0)
        parity_check_success = fec_candidate.get("parity_check_success", False)
        path_metric = fec_candidate.get("path_metric", 0.0)
        decoder_metrics = fec_candidate.get("decoder_metrics", {}) or {}
    else:
        return SyndromeResult(check_state=EvidenceCheckState.NOT_TESTED)

    if family in ("none", "uncoded"):
        return SyndromeResult(
            syndrome_available=False,
            syndrome_valid=False,
            check_state=EvidenceCheckState.NOT_TESTED,
            normalized_syndrome_score=0.5,
        )

    syndrome_available = True
    syndrome_valid = False
    normalized_score = 0.0

    if family == "reed_solomon":
        # RS syndrome weight should be 0.0 after successful error correction
        syndrome_weight_val = float(syndrome_weight)
        final_synd_weight = float(decoder_metrics.get("final_syndrome_weight", syndrome_weight_val))
        initial_synd_weight = float(decoder_metrics.get("initial_syndrome_weight", final_synd_weight))

        if decoder_success and final_synd_weight == 0.0:
            syndrome_valid = True
            normalized_score = 1.0
            check_state = EvidenceCheckState.PASS
        elif not decoder_success:
            syndrome_valid = False
            normalized_score = 0.0
            check_state = EvidenceCheckState.FAIL
        else:
            syndrome_valid = False
            normalized_score = max(0.0, 1.0 - final_synd_weight / 10.0)
            check_state = EvidenceCheckState.FAIL

        return SyndromeResult(
            syndrome_available=True,
            syndrome_valid=syndrome_valid,
            initial_syndrome_weight=initial_synd_weight,
            final_syndrome_weight=final_synd_weight,
            normalized_syndrome_score=normalized_score,
            check_state=check_state,
        )

    elif family == "ldpc":
        # LDPC parity checks satisfied (syndrome = 0)
        pc_success = bool(parity_check_success or decoder_metrics.get("parity_check_success", False))
        iterations = float(decoder_metrics.get("iterations", 0))
        max_iters = float(decoder_metrics.get("max_iterations", 20))

        if pc_success and decoder_success:
            syndrome_valid = True
            # Converging in fewer iterations gives a higher score
            speed_bonus = max(0.0, 1.0 - (iterations / max(1.0, max_iters))) * 0.2
            normalized_score = min(1.0, 0.8 + speed_bonus)
            check_state = EvidenceCheckState.PASS
        elif not decoder_success or not pc_success:
            syndrome_valid = False
            normalized_score = 0.0
            check_state = EvidenceCheckState.FAIL
        else:
            syndrome_valid = False
            normalized_score = 0.3
            check_state = EvidenceCheckState.FAIL

        return SyndromeResult(
            syndrome_available=True,
            syndrome_valid=syndrome_valid,
            initial_syndrome_weight=1.0 if not pc_success else 0.0,
            final_syndrome_weight=0.0 if pc_success else 1.0,
            normalized_syndrome_score=normalized_score,
            check_state=check_state,
        )

    elif family == "convolutional":
        # Viterbi normalized path metric & termination
        norm_pm = float(decoder_metrics.get("normalized_path_metric", path_metric))
        term_ok = bool(decoder_metrics.get("termination_valid", True))

        if decoder_success and norm_pm < 0.25 and term_ok:
            syndrome_valid = True
            normalized_score = max(0.0, 1.0 - norm_pm * 2.0)
            check_state = EvidenceCheckState.PASS
        elif decoder_success:
            syndrome_valid = False
            normalized_score = max(0.0, 1.0 - norm_pm)
            check_state = EvidenceCheckState.NOT_TESTED
        else:
            syndrome_valid = False
            normalized_score = 0.0
            check_state = EvidenceCheckState.FAIL

        return SyndromeResult(
            syndrome_available=True,
            syndrome_valid=syndrome_valid,
            initial_syndrome_weight=norm_pm,
            final_syndrome_weight=norm_pm,
            normalized_syndrome_score=normalized_score,
            check_state=check_state,
        )

    return SyndromeResult(
        syndrome_available=True,
        syndrome_valid=bool(decoder_success),
        normalized_syndrome_score=1.0 if decoder_success else 0.0,
        check_state=EvidenceCheckState.PASS if decoder_success else EvidenceCheckState.FAIL,
    )
