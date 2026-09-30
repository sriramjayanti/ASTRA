"""
Unit tests for confidence scoring, decomposition, margins, penalties, and gating.
Tests:
- TEST 7: Confidence bounded [0, 1]
- TEST 13: Contradiction penalty application
- TEST 14: Candidate margin effect on ranking confidence
- TEST 21: Payload vs semantic confidence separation
"""

import pytest
from astra_explainability.src.models import (
    FieldExplanation, CandidateExplanation, Contradiction,
    ContradictionSeverity, AstraStatus
)
from astra_explainability.src.confidence import ConfidenceEngine
from astra_explainability.src.statuses import StatusEngine
from astra_explainability.src.inference import ExplainabilityEngine
from astra_explainability.src.utils import (
    create_synthetic_perfect_record, create_tied_candidates_record,
    create_contradictory_record
)


def test_7_confidence_bounded():
    cfg = {
        "confidence": {
            "modulation_weight": 0.10, "symbol_rate_weight": 0.10,
            "synchronization_weight": 0.10, "demodulation_weight": 0.10,
            "fec_weight": 0.15, "validation_weight": 0.25,
            "pipeline_ranking_weight": 0.15, "structure_weight": 0.05
        },
        "contradiction": {
            "low_penalty": 0.02, "medium_penalty": 0.05,
            "high_penalty": 0.15, "critical_penalty": 0.30
        }
    }
    status_engine = StatusEngine()
    engine = ConfidenceEngine(cfg, status_engine)

    # Edge test: all 1.0 confidences
    fields = {
        "modulation": FieldExplanation("modulation", "QPSK", AstraStatus.CONFIRMED, 1.0),
        "validation": FieldExplanation("validation", "Pass", AstraStatus.CONFIRMED, 1.0)
    }
    candidates = [CandidateExplanation("p1", 0, 1.0, 1.0, "QPSK", 9600.0, "None", "None", True)]

    overall, breakdown = engine.calculate_overall_confidence(
        field_explanations=fields,
        candidate_explanations=candidates,
        contradictions=[],
        missing_evidence=[],
        all_evidence=[]
    )
    assert 0.0 <= overall <= 1.0

    # Edge test: high penalties
    contras = [
        Contradiction("c1", ["modulation"], "ev1", "ev2", ContradictionSeverity.CRITICAL, "Conflict", False),
        Contradiction("c2", ["fec"], "ev3", "ev4", ContradictionSeverity.HIGH, "Conflict", False),
    ]
    overall_low, _ = engine.calculate_overall_confidence(
        field_explanations=fields,
        candidate_explanations=candidates,
        contradictions=contras,
        missing_evidence=[],
        all_evidence=[]
    )
    assert 0.0 <= overall_low <= 1.0
    assert overall_low < overall


def test_13_contradiction_penalties():
    engine = ExplainabilityEngine()
    perf_record = create_synthetic_perfect_record()
    res_perf = engine.explain(perf_record)

    contra_record = create_contradictory_record()
    res_contra = engine.explain(contra_record)

    assert len(res_contra.contradictions) > 0
    assert res_contra.overall_confidence < res_perf.overall_confidence
    assert res_contra.confidence_breakdown.contradiction_penalties > 0.0


def test_14_candidate_margin_effect():
    engine = ExplainabilityEngine()

    # Clear winner record: candidate 1 is 0.965 vs candidate 2 is 0.380 (margin 0.585)
    perf_record = create_synthetic_perfect_record()
    res_clear = engine.explain(perf_record)

    # Tied record: candidate 1 is 0.810 vs candidate 2 is 0.805 (margin 0.005)
    tied_record = create_tied_candidates_record()
    res_tied = engine.explain(tied_record)

    assert res_clear.confidence_breakdown.ranking_confidence > res_tied.confidence_breakdown.ranking_confidence


def test_21_payload_semantic_confidence_separation():
    engine = ExplainabilityEngine()
    record = create_synthetic_perfect_record()
    res = engine.explain(record)

    assert "payload" in res.field_explanations
    assert "text_representation" in res.field_explanations

    pay_exp = res.field_explanations["payload"]
    txt_exp = res.field_explanations["text_representation"]

    # Byte recovery is CONFIRMED from CRC & FEC
    assert pay_exp.status == AstraStatus.CONFIRMED
    # Semantic text interpretation is ESTIMATED because protocol is blind/undeclared
    assert txt_exp.status == AstraStatus.ESTIMATED
