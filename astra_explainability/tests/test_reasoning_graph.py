"""
Unit tests for Reasoning Graph generation (nodes, edges, DAG properties).
Tests:
- TEST 18: Reasoning graph nodes
- TEST 19: Reasoning graph edges
"""

import pytest
from astra_explainability.src.models import NodeType, EdgeType
from astra_explainability.src.reasoning_graph import ReasoningGraphBuilder
from astra_explainability.src.evidence import EvidenceExtractor
from astra_explainability.src.contradictions import ContradictionAnalyzer
from astra_explainability.src.field_reasoners import ModulationReasoner, ValidationReasoner
from astra_explainability.src.statuses import StatusEngine
from astra_explainability.src.utils import create_synthetic_perfect_record


def test_18_and_19_reasoning_graph():
    builder = ReasoningGraphBuilder()
    extractor = EvidenceExtractor()
    analyzer = ContradictionAnalyzer()
    status_engine = StatusEngine()

    record = create_synthetic_perfect_record()
    evidence = extractor.extract_all(record)
    contras = analyzer.analyze(record, evidence)

    mod_reasoner = ModulationReasoner(status_engine)
    val_reasoner = ValidationReasoner(status_engine)

    fields = {
        "modulation": mod_reasoner.reason(record, evidence, contras),
        "validation": val_reasoner.reason(record, evidence, contras)
    }

    graph = builder.build_graph(fields, evidence, contras)

    # Check node types
    node_types = {n.node_type for n in graph.nodes}
    assert NodeType.STAGE in node_types
    assert NodeType.CLAIM in node_types
    assert NodeType.EVIDENCE in node_types

    # Check edge types
    edge_types = {e.edge_type for e in graph.edges}
    assert EdgeType.DERIVED_FROM in edge_types
    assert EdgeType.SUPPORTS in edge_types

    # Ensure claims are present
    claim_nodes = [n for n in graph.nodes if n.node_type == NodeType.CLAIM]
    claim_ids = [c.node_id for c in claim_nodes]
    assert "claim_modulation" in claim_ids
    assert "claim_validation" in claim_ids

    # Ensure stages 1-14 are present
    stage_nodes = [n for n in graph.nodes if n.node_type == NodeType.STAGE]
    assert len(stage_nodes) >= 14
