"""
Comprehensive End-to-End Integration and Master System Tests.
Tests:
- TEST 24: Model version provenance
- TEST 25: Config provenance
- TEST 26: Deterministic output
- TEST 27: Compact human summary
- TEST 28: Technical summary
- TEST 29: GUI summary & cards
- TEST 30: Full Stage 1–14 integration object
- Perfect "hi hello" recovery verification
- Low-SNR scenario verification
- Wrong early model override verification
"""

import json
import pytest
from astra_explainability.src.inference import ExplainabilityEngine
from astra_explainability.src.models import AstraStatus
from astra_explainability.src.utils import (
    create_synthetic_perfect_record, create_low_snr_record,
    create_wrong_top1_model_record
)


def test_30_full_integration_hi_hello():
    """
    MASTER INTEGRATION TEST:
    Recovers QPSK, 9600 baud, Block 16x16, Conv K7 R1/2, 512-bit frame, and 'hi hello' payload.
    """
    engine = ExplainabilityEngine()
    record = create_synthetic_perfect_record()

    res = engine.explain(record)

    # 1. Overall Status & Confidence
    assert res.overall_status == AstraStatus.CONFIRMED
    assert res.overall_confidence >= 0.85

    # 2. Field Explanations
    fields = res.field_explanations
    assert fields["modulation"].value == "QPSK"
    assert fields["modulation"].status == AstraStatus.CONFIRMED

    assert fields["symbol_rate"].value == 9600.0
    assert fields["symbol_rate"].status in (AstraStatus.CONFIRMED, AstraStatus.ESTIMATED)

    assert fields["synchronization"].status == AstraStatus.CONFIRMED

    assert "Block 16x16" in str(fields["interleaver"].value)
    assert fields["interleaver"].status in (AstraStatus.CONFIRMED, AstraStatus.ESTIMATED)

    assert "Convolutional K7 R1/2" in str(fields["fec"].value)
    assert fields["fec"].status == AstraStatus.CONFIRMED

    assert fields["validation"].status == AstraStatus.CONFIRMED

    # Payload bytes CONFIRMED, text representation ESTIMATED, protocol UNKNOWN
    pay_val = fields["payload"].value
    assert pay_val["hex"] == "68692068656c6c6f"
    assert fields["payload"].status == AstraStatus.CONFIRMED

    assert fields["text_representation"].value == "hi hello"
    assert fields["text_representation"].status == AstraStatus.ESTIMATED

    assert fields["protocol"].status == AstraStatus.UNKNOWN


def test_26_deterministic_output():
    """Identical input and config must yield identical outputs."""
    engine = ExplainabilityEngine()
    record = create_synthetic_perfect_record()

    res1 = engine.explain(record)
    res2 = engine.explain(record)

    assert res1.overall_status == res2.overall_status
    assert res1.overall_confidence == res2.overall_confidence
    assert res1.human_summary == res2.human_summary
    assert res1.technical_summary == res2.technical_summary
    assert len(res1.reasoning_graph.nodes) == len(res2.reasoning_graph.nodes)


def test_24_and_25_provenance():
    """Provenance tracking for models and config hash."""
    engine = ExplainabilityEngine()
    record = create_synthetic_perfect_record()

    res = engine.explain(record)
    prov = res.processing_provenance

    assert "model_versions" in prov
    assert "config_hash" in prov
    assert prov["config_hash"].startswith("sha256:")
    assert "resnet_1d" in prov["model_versions"]
    assert "pipeline_xgboost_scorer" in prov["model_versions"]


def test_27_28_29_summaries():
    """Human summary, technical summary, and GUI cards."""
    engine = ExplainabilityEngine()
    record = create_synthetic_perfect_record()
    res = engine.explain(record)

    # Human summary contains key recovered entities
    h_sum = res.human_summary
    assert "QPSK" in h_sum
    assert "9600" in h_sum
    assert "hi hello" in h_sum

    # Technical summary contains engineering sections
    t_sum = res.technical_summary
    assert "=== ASTRA STAGE 15: TECHNICAL EXPLAINABILITY REPORT ===" in t_sum
    assert "Overall Status: CONFIRMED" in t_sum

    # GUI Summary cards
    gui = res.gui_summary
    assert gui.overall_status == AstraStatus.CONFIRMED
    assert len(gui.cards) >= 5
    card_names = [c.field_name for c in gui.cards]
    assert "modulation" in card_names
    assert "payload" in card_names


def test_low_snr_scenario():
    """Low SNR scenario where downstream validation reinforces early uncertainty."""
    engine = ExplainabilityEngine()
    record = create_low_snr_record()
    res = engine.explain(record)

    assert res.overall_status in (AstraStatus.CONFIRMED, AstraStatus.ESTIMATED)
    # Downstream validation is strong (6/8 CRC checks)
    assert res.field_explanations["validation"].status == AstraStatus.CONFIRMED


def test_wrong_early_model_override():
    """Downstream deterministic validation must override early model disagreement."""
    engine = ExplainabilityEngine()
    record = create_wrong_top1_model_record()
    res = engine.explain(record)

    # Even though Stage 3 fusion predicted 8PSK, Stage 11 pipeline winner & CRC confirmed QPSK!
    mod_exp = res.field_explanations["modulation"]
    assert mod_exp.value == "QPSK"
    # Secondary candidate 8PSK noted in contradictions
    assert any("8PSK" in c for c in mod_exp.contradicting_evidence)
