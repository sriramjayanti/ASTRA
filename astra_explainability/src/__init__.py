"""
ASTRA Stage 15: Explainability & Confidence Reasoning Engine.
"""

from .models import (
    AstraStatus,
    EvidenceCategory,
    EvidenceStrength,
    EvidenceDirection,
    IndependenceGroup,
    ContradictionSeverity,
    NodeType,
    EdgeType,
    EvidenceItem,
    EvidenceGroup,
    Contradiction,
    MissingEvidence,
    FieldExplanation,
    CandidateExplanation,
    ConfidenceBreakdown,
    ReasoningNode,
    ReasoningEdge,
    ReasoningGraph,
    GuiCard,
    GuiSummary,
    ExplainabilityResult
)

from .inference import ExplainabilityEngine
from .evidence import EvidenceExtractor
from .evidence_groups import EvidenceGrouper
from .contradictions import ContradictionAnalyzer
from .missing_evidence import MissingEvidenceAnalyzer
from .candidate_comparison import CandidateComparator
from .confidence import ConfidenceEngine
from .statuses import StatusEngine
from .reasoning_graph import ReasoningGraphBuilder
from .summaries import SummaryGenerator
from .provenance import ProvenanceTracker

__all__ = [
    "ExplainabilityEngine",
    "AstraStatus",
    "EvidenceCategory",
    "EvidenceStrength",
    "EvidenceDirection",
    "IndependenceGroup",
    "ContradictionSeverity",
    "NodeType",
    "EdgeType",
    "EvidenceItem",
    "EvidenceGroup",
    "Contradiction",
    "MissingEvidence",
    "FieldExplanation",
    "CandidateExplanation",
    "ConfidenceBreakdown",
    "ReasoningNode",
    "ReasoningEdge",
    "ReasoningGraph",
    "GuiCard",
    "GuiSummary",
    "ExplainabilityResult",
    "EvidenceExtractor",
    "EvidenceGrouper",
    "ContradictionAnalyzer",
    "MissingEvidenceAnalyzer",
    "CandidateComparator",
    "ConfidenceEngine",
    "StatusEngine",
    "ReasoningGraphBuilder",
    "SummaryGenerator",
    "ProvenanceTracker"
]
