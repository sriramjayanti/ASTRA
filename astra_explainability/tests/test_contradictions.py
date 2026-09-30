"""
Unit tests for contradiction detection, severity assignment, and missing evidence.
Tests:
- TEST 5: Contradictory evidence
- TEST 6: Missing evidence auditing
- TEST 13: Contradiction detection across pipeline
"""

import pytest
from astra_explainability.src.contradictions import ContradictionAnalyzer
from astra_explainability.src.missing_evidence import MissingEvidenceAnalyzer
from astra_explainability.src.evidence import EvidenceExtractor
from astra_explainability.src.models import ContradictionSeverity
from astra_explainability.src.utils import (
    create_synthetic_perfect_record, create_contradictory_record,
    create_missing_crc_record, create_user_override_record
)


def test_6_missing_evidence():
    analyzer = MissingEvidenceAnalyzer()

    # Missing CRC record
    record = create_missing_crc_record()
    missing = analyzer.analyze(record)

    missing_types = [m.evidence_type for m in missing]
    assert "CRC Validation" in missing_types
    assert "Protocol Profile" in missing_types

    # Penalties are bounded
    for m in missing:
        assert 0.0 <= m.penalty <= 0.15


def test_contradiction_detection():
    analyzer = ContradictionAnalyzer()
    extractor = EvidenceExtractor()

    # Perfect record has zero severe contradictions
    perf_record = create_synthetic_perfect_record()
    ev_perf = extractor.extract_all(perf_record)
    contras_perf = analyzer.analyze(perf_record, ev_perf)
    high_perf = [c for c in contras_perf if c.severity in (ContradictionSeverity.HIGH, ContradictionSeverity.CRITICAL)]
    assert len(high_perf) == 0

    # Contradictory record: strong sync/demod but CRC 0/12
    contra_record = create_contradictory_record()
    ev_contra = extractor.extract_all(contra_record)
    contras = analyzer.analyze(contra_record, ev_contra)

    assert len(contras) > 0
    crc_contras = [c for c in contras if "validation" in c.fields_involved or "crc" in c.description.lower()]
    assert len(crc_contras) > 0
    assert crc_contras[0].severity in (ContradictionSeverity.HIGH, ContradictionSeverity.CRITICAL)


def test_user_override_contradiction():
    analyzer = ContradictionAnalyzer()
    extractor = EvidenceExtractor()

    # User overrides modulation to 16QAM while models support QPSK
    override_record = create_user_override_record()
    ev_override = extractor.extract_all(override_record)
    contras = analyzer.analyze(override_record, ev_override)

    override_contras = [c for c in contras if "override" in c.description.lower()]
    assert len(override_contras) > 0
    assert "modulation" in override_contras[0].fields_involved
