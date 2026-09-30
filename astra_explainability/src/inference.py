"""
ASTRA Stage 15: Main Explainability & Confidence Reasoning Engine.

Orchestrates all explainability submodules across Stages 1-14:
- Atomic evidence extraction
- Independence clustering
- Contradiction & missing evidence auditing
- Candidate comparison & why-lost attribution
- Field-specific reasoning and status assignment
- Confidence synthesis and decomposition
- Machine-readable reasoning graph construction
- Multi-tier summaries (human, technical, GUI)
- Version and configuration provenance
"""

import os
import yaml
from typing import Dict, Any, List, Optional

from .models import (
    ExplainabilityResult, FieldExplanation, CandidateExplanation,
    ConfidenceBreakdown, ReasoningGraph, GuiSummary, Contradiction,
    MissingEvidence, AstraStatus
)
from .evidence import EvidenceExtractor
from .evidence_groups import EvidenceGrouper
from .contradictions import ContradictionAnalyzer
from .missing_evidence import MissingEvidenceAnalyzer
from .candidate_comparison import CandidateComparator
from .field_reasoners import (
    ModulationReasoner, SymbolRateReasoner, SynchronizationReasoner,
    DemodulationReasoner, InterleaverReasoner, FECReasoner,
    ValidationReasoner, FrameReasoner, PayloadReasoner, ProtocolReasoner
)
from .confidence import ConfidenceEngine
from .statuses import StatusEngine
from .reasoning_graph import ReasoningGraphBuilder
from .summaries import SummaryGenerator
from .provenance import ProvenanceTracker
from .validators import InputValidator


class ExplainabilityEngine:
    """
    Main ASTRA Stage 15 Engine.
    Transforms raw multi-stage evidence into explainable, traceable, auditable reasoning.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Loads configuration and instantiates submodules."""
        self.config = self._load_config(config_path)

        self.evidence_extractor = EvidenceExtractor()
        self.evidence_grouper = EvidenceGrouper()
        self.contradiction_analyzer = ContradictionAnalyzer()
        self.missing_analyzer = MissingEvidenceAnalyzer()
        self.candidate_comparator = CandidateComparator()
        self.status_engine = StatusEngine(self.config.get("statuses", {}))
        self.confidence_engine = ConfidenceEngine(self.config, self.status_engine)

        # Field reasoners
        self.field_reasoners = {
            "modulation": ModulationReasoner(self.status_engine),
            "symbol_rate": SymbolRateReasoner(self.status_engine),
            "synchronization": SynchronizationReasoner(self.status_engine),
            "demodulation": DemodulationReasoner(self.status_engine),
            "interleaver": InterleaverReasoner(self.status_engine),
            "fec": FECReasoner(self.status_engine),
            "validation": ValidationReasoner(self.status_engine),
            "frame_structure": FrameReasoner(self.status_engine),
            "payload": PayloadReasoner(self.status_engine),
            "protocol": ProtocolReasoner(self.status_engine),
        }

        self.graph_builder = ReasoningGraphBuilder()
        self.summary_generator = SummaryGenerator()
        self.provenance_tracker = ProvenanceTracker()

    def explain(self, analysis_record: Dict[str, Any]) -> ExplainabilityResult:
        """
        Executes end-to-end explainability synthesis on a complete Stage 1-14 record.
        """
        # 1. Validate inputs
        validation_warnings = InputValidator.validate_record(analysis_record)

        signal_id = analysis_record.get("signal_id", "unknown_signal")
        stage_11 = analysis_record.get("stage_11_ranking", {})
        best_pipeline_id = stage_11.get("best_pipeline_id", "unknown_pipeline")

        # 2. Extract atomic evidence items
        all_evidence = self.evidence_extractor.extract_all(analysis_record)

        # 3. Detect contradictions
        contradictions = self.contradiction_analyzer.analyze(analysis_record, all_evidence)

        # 4. Audit missing evidence
        missing_evidence = self.missing_analyzer.analyze(analysis_record)

        # 5. Field-specific reasoning and status assignment
        field_explanations: Dict[str, FieldExplanation] = {}
        for field_name, reasoner in self.field_reasoners.items():
            field_contras = [c for c in contradictions if field_name in c.fields_involved]
            field_exp = reasoner.reason(analysis_record, all_evidence, field_contras)
            field_explanations[field_name] = field_exp

        # 5b. Add separate text representation explanation if text payload exists
        payload_exp = field_explanations.get("payload")
        if payload_exp and isinstance(payload_exp.value, dict) and payload_exp.value.get("text"):
            text_val = payload_exp.value.get("text")
            is_utf8 = payload_exp.value.get("is_utf8", False)
            # Text semantic interpretation is ESTIMATED unless verified by protocol
            txt_status = AstraStatus.ESTIMATED if is_utf8 else AstraStatus.POSSIBLE
            txt_conf = min(payload_exp.confidence_score, 0.93)
            field_explanations["text_representation"] = FieldExplanation(
                field_name="text_representation",
                value=text_val,
                status=txt_status,
                confidence_score=txt_conf,
                confidence_type="ASTRA Confidence",
                supporting_evidence=["Decoded bytes form valid UTF-8 character sequence"],
                source_stages=["Stage 14: Header / Payload Explorer"],
                status_reason=f"{txt_status.value} because bytes map cleanly to UTF-8 text."
            )

        # 6. Compare candidate hypotheses and explain why losers lost
        candidates = stage_11.get("candidates", [])
        top_k = self.config.get("alternatives", {}).get("top_k", 3)
        candidate_explanations = self.candidate_comparator.compare_candidates(candidates, top_k=top_k)

        # 7. Synthesize overall confidence and decomposition
        overall_conf, breakdown = self.confidence_engine.calculate_overall_confidence(
            field_explanations=field_explanations,
            candidate_explanations=candidate_explanations,
            contradictions=contradictions,
            missing_evidence=missing_evidence,
            all_evidence=all_evidence
        )

        # 8. Determine overall pipeline status
        overall_status = self._determine_overall_status(
            field_explanations, overall_conf, contradictions
        )

        # 9. Build machine-readable reasoning DAG
        reasoning_graph = self.graph_builder.build_graph(
            field_explanations, all_evidence, contradictions
        )

        # 10. Generate summaries (human, technical, GUI)
        human_sum = self.summary_generator.generate_human_summary(
            overall_status, overall_conf, field_explanations, contradictions
        )
        tech_sum = self.summary_generator.generate_technical_summary(
            overall_status, overall_conf, field_explanations, candidate_explanations,
            contradictions, missing_evidence
        )
        gui_sum = self.summary_generator.generate_gui_summary(
            overall_status, overall_conf, field_explanations, candidate_explanations,
            contradictions
        )

        # 11. Provenance metadata
        provenance = self.provenance_tracker.get_provenance_record(
            self.config, analysis_record.get("model_versions")
        )

        # Strong evidence highlights
        strong_evidence_desc = [
            ev.description for ev in all_evidence
            if ev.strength.value in ("STRONG", "MODERATE") and ev.direction.value == "SUPPORT"
        ][:8]

        # Alternative hypotheses summary strings
        alt_summaries = [
            f"#{c.rank} {c.pipeline_id} (Score: {c.score:.3f}): {c.why_lost or 'Winning candidate'}"
            for c in candidate_explanations
        ]

        return ExplainabilityResult(
            signal_id=signal_id,
            best_pipeline_id=best_pipeline_id,
            overall_status=overall_status,
            overall_confidence=round(overall_conf, 4),
            field_explanations=field_explanations,
            candidate_comparison=candidate_explanations,
            evidence_summary=all_evidence,
            contradictions=contradictions,
            missing_evidence=missing_evidence,
            uncertainty_sources=[m.impact_description for m in missing_evidence],
            alternative_hypotheses=alt_summaries,
            reasoning_graph=reasoning_graph,
            human_summary=human_sum,
            technical_summary=tech_sum,
            gui_summary=gui_sum,
            confidence_breakdown=breakdown,
            model_versions=provenance["model_versions"],
            processing_provenance=provenance
        )

    def _determine_overall_status(
        self,
        fields: Dict[str, FieldExplanation],
        overall_conf: float,
        contradictions: List[Contradiction]
    ) -> AstraStatus:
        """Determines overarching signal recovery status with gating."""
        # Critical contradictions force UNKNOWN or POSSIBLE
        critical_c = [c for c in contradictions if c.severity.value == "CRITICAL"]
        if critical_c:
            return AstraStatus.UNKNOWN

        # If payload or validation is confirmed, and confidence >= 0.85
        val_status = fields.get("validation", FieldExplanation("", None, AstraStatus.UNKNOWN, 0.0)).status
        mod_status = fields.get("modulation", FieldExplanation("", None, AstraStatus.UNKNOWN, 0.0)).status

        if overall_conf >= 0.85 and val_status == AstraStatus.CONFIRMED and mod_status in (AstraStatus.CONFIRMED, AstraStatus.ESTIMATED):
            return AstraStatus.CONFIRMED
        elif overall_conf >= 0.60:
            return AstraStatus.ESTIMATED
        elif overall_conf >= 0.35:
            return AstraStatus.POSSIBLE
        return AstraStatus.UNKNOWN

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Loads YAML config or falls back to sane defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        # Default path
        default_cfg = os.path.join(
            os.path.dirname(__file__), "..", "configs", "explainability_config.yaml"
        )
        if os.path.exists(default_cfg):
            with open(default_cfg, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        return {
            "version": "1.0.0",
            "statuses": {
                "confirmed_threshold": 0.85,
                "estimated_threshold": 0.60,
                "possible_threshold": 0.35,
                "confirmed_min_strong_groups": 2
            },
            "confidence": {
                "modulation_weight": 0.10,
                "symbol_rate_weight": 0.10,
                "synchronization_weight": 0.10,
                "demodulation_weight": 0.10,
                "fec_weight": 0.15,
                "validation_weight": 0.25,
                "pipeline_ranking_weight": 0.15,
                "structure_weight": 0.05
            },
            "contradiction": {
                "low_penalty": 0.02,
                "medium_penalty": 0.05,
                "high_penalty": 0.15,
                "critical_penalty": 0.30
            },
            "alternatives": {
                "top_k": 3
            }
        }
