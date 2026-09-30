"""
test_syndrome.py
Unit tests for Stage 9 FEC syndrome and telemetry normalization.
"""

import pytest
from astra_validation.src.syndrome import normalize_syndrome_evidence
from astra_validation.src.models import EvidenceCheckState


def test_rs_syndrome_normalization_pass():
    mock_rs = {
        "fec_family": "reed_solomon",
        "decoder_success": True,
        "syndrome_weight": 0.0,
        "decoder_metrics": {"initial_syndrome_weight": 4.0, "final_syndrome_weight": 0.0},
    }
    res = normalize_syndrome_evidence(mock_rs)
    assert res.syndrome_available is True
    assert res.syndrome_valid is True
    assert res.normalized_syndrome_score == 1.0
    assert res.check_state == EvidenceCheckState.PASS


def test_rs_syndrome_normalization_fail():
    mock_rs = {
        "fec_family": "reed_solomon",
        "decoder_success": False,
        "syndrome_weight": 5.0,
        "decoder_metrics": {"initial_syndrome_weight": 5.0, "final_syndrome_weight": 5.0},
    }
    res = normalize_syndrome_evidence(mock_rs)
    assert res.syndrome_available is True
    assert res.syndrome_valid is False
    assert res.check_state == EvidenceCheckState.FAIL


def test_ldpc_parity_success():
    mock_ldpc = {
        "fec_family": "ldpc",
        "decoder_success": True,
        "parity_check_success": True,
        "decoder_metrics": {"iterations": 3, "max_iterations": 20},
    }
    res = normalize_syndrome_evidence(mock_ldpc)
    assert res.syndrome_valid is True
    assert res.normalized_syndrome_score >= 0.8
    assert res.check_state == EvidenceCheckState.PASS


def test_viterbi_metric_normalization():
    mock_viterbi = {
        "fec_family": "convolutional",
        "decoder_success": True,
        "path_metric": 0.05,
        "decoder_metrics": {"normalized_path_metric": 0.05, "termination_valid": True},
    }
    res = normalize_syndrome_evidence(mock_viterbi)
    assert res.syndrome_valid is True
    assert res.normalized_syndrome_score >= 0.8
    assert res.check_state == EvidenceCheckState.PASS
