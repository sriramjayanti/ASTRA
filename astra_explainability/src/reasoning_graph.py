"""
ASTRA Stage 15: Reasoning Graph Builder.

Constructs an auditable DAG of claims, stages, evidence items, and contradictions.
"""

from typing import Dict, List, Any
from .models import (
    ReasoningGraph, ReasoningNode, ReasoningEdge,
    NodeType, EdgeType, FieldExplanation, EvidenceItem, Contradiction
)


class ReasoningGraphBuilder:
    """Builds machine-readable explanation graph for GUI and audit trails."""

    def build_graph(
        self,
        field_explanations: Dict[str, FieldExplanation],
        all_evidence: List[EvidenceItem],
        contradictions: List[Contradiction]
    ) -> ReasoningGraph:
        """
        Builds the explanation DAG.
        Nodes:
        - Claims (field values and statuses)
        - Stages (Stage 1 to 14)
        - Evidence items
        - Contradictions

        Edges:
        - derived_from (Claim -> Stage, Evidence -> Stage)
        - supports (Evidence -> Claim)
        - contradicts (Evidence -> Claim, or Contradiction -> Claim)
        - depends_on (Hierarchical claim dependency, e.g. Payload depends_on Frame)
        """
        nodes: List[ReasoningNode] = []
        edges: List[ReasoningEdge] = []
        node_ids = set()

        def add_node(node: ReasoningNode):
            if node.node_id not in node_ids:
                nodes.append(node)
                node_ids.add(node.node_id)

        # 1. Add Stage Nodes
        stage_names = {
            1: "Signal Ingestion & RF",
            2: "Feature Extraction & DSP",
            3: "Modulation Neural Models",
            4: "Constellation Analysis",
            5: "Candidate Engine",
            6: "Synchronization",
            7: "Demodulation",
            8: "Interleaver Testing",
            9: "FEC Testing",
            10: "Validation Engine",
            11: "Pipeline Scorer",
            12: "Bitstream Intelligence",
            13: "CNN + Transformer Structure",
            14: "Header & Payload Explorer",
            15: "Explainability Engine"
        }
        for s_num, s_desc in stage_names.items():
            s_id = f"stage_{s_num}"
            add_node(ReasoningNode(
                node_id=s_id,
                node_type=NodeType.STAGE,
                label=f"Stage {s_num}: {s_desc}",
                properties={"stage_number": s_num}
            ))

        # 2. Add Claim Nodes
        for field_name, f_exp in field_explanations.items():
            claim_id = f"claim_{field_name}"
            add_node(ReasoningNode(
                node_id=claim_id,
                node_type=NodeType.CLAIM,
                label=f"{field_name.upper()} = {f_exp.value} ({f_exp.status.value})",
                status=f_exp.status,
                confidence_score=f_exp.confidence_score,
                properties={
                    "field": field_name,
                    "value": str(f_exp.value),
                    "status_reason": f_exp.status_reason
                }
            ))

            # Connect Claim to its source stages
            for src_stage in f_exp.source_stages:
                s_id = f"stage_{src_stage}"
                if s_id in node_ids:
                    edges.append(ReasoningEdge(
                        source_id=claim_id,
                        target_id=s_id,
                        edge_type=EdgeType.DERIVED_FROM,
                        label="evaluated by"
                    ))

        # 3. Add Evidence Nodes & link to Stages and Claims
        for ev in all_evidence:
            ev_node_id = f"ev_{ev.evidence_id}"
            add_node(ReasoningNode(
                node_id=ev_node_id,
                node_type=NodeType.EVIDENCE,
                label=f"[{ev.strength.value}] {ev.description[:45]}...",
                confidence_score=ev.normalized_score,
                properties={
                    "stage": ev.stage,
                    "category": ev.category.value,
                    "independence_group": ev.independence_group.value,
                    "direction": ev.direction.value,
                    "value": str(ev.value)
                }
            ))

            # Connect Evidence -> Stage
            s_num = None
            if isinstance(ev.stage, int):
                s_num = ev.stage
            elif isinstance(ev.stage, str) and "Stage " in ev.stage:
                parts = ev.stage.split("Stage ")[1].split(":")[0].strip()
                if parts.isdigit():
                    s_num = int(parts)

            if s_num:
                s_id = f"stage_{s_num}"
                if s_id in node_ids:
                    edges.append(ReasoningEdge(
                        source_id=ev_node_id,
                        target_id=s_id,
                        edge_type=EdgeType.DERIVED_FROM,
                        label="measured at"
                    ))

            # Connect Evidence -> Claims
            # Map evidence categories/descriptions to claims
            self._link_evidence_to_claims(ev, ev_node_id, field_explanations, edges)

        # 4. Add Contradiction Nodes & link to Claims
        for c in contradictions:
            c_node_id = f"contra_{c.contradiction_id}"
            add_node(ReasoningNode(
                node_id=c_node_id,
                node_type=NodeType.CONTRADICTION,
                label=f"Contradiction [{c.severity.value}]: {c.description[:40]}...",
                properties={
                    "severity": c.severity.value,
                    "fields": c.fields_involved,
                    "resolved": c.resolution_status
                }
            ))

            for field_name in c.fields_involved:
                claim_id = f"claim_{field_name}"
                if claim_id in node_ids:
                    edges.append(ReasoningEdge(
                        source_id=c_node_id,
                        target_id=claim_id,
                        edge_type=EdgeType.CONTRADICTS,
                        label=f"{c.severity.value} conflict"
                    ))

        # 5. Add Claim Dependencies (Hierarchical domain logic)
        # Payload -> Frame -> FEC -> Interleaver -> Demod -> Sync -> Modulation / Symbol Rate
        claim_dependencies = [
            ("claim_payload", "claim_frame_structure"),
            ("claim_payload", "claim_fec"),
            ("claim_text_representation", "claim_payload"),
            ("claim_protocol", "claim_payload"),
            ("claim_frame_structure", "claim_validation"),
            ("claim_frame_structure", "claim_fec"),
            ("claim_validation", "claim_fec"),
            ("claim_fec", "claim_interleaver"),
            ("claim_interleaver", "claim_demodulation"),
            ("claim_demodulation", "claim_synchronization"),
            ("claim_synchronization", "claim_modulation"),
            ("claim_synchronization", "claim_symbol_rate"),
        ]

        for src, tgt in claim_dependencies:
            if src in node_ids and tgt in node_ids:
                edges.append(ReasoningEdge(
                    source_id=src,
                    target_id=tgt,
                    edge_type=EdgeType.DEPENDS_ON,
                    label="pipeline dependency"
                ))

        return ReasoningGraph(nodes=nodes, edges=edges)

    def _link_evidence_to_claims(
        self,
        ev: EvidenceItem,
        ev_node_id: str,
        field_explanations: Dict[str, FieldExplanation],
        edges: List[ReasoningEdge]
    ):
        """Map evidence to field claims with support/contradict edges."""
        cat = ev.category.value
        edge_type = EdgeType.SUPPORTS if ev.direction.value == "SUPPORT" else EdgeType.CONTRADICTS

        target_claim_keys = []
        if "MODEL" in cat or "CONSTELLATION" in cat or ev.independence_group.value == "MODULATION_MODELS":
            target_claim_keys.append("modulation")
        if "SYNC" in cat or ev.independence_group.value in ("SYNC", "DSP_SIGNAL_STRUCTURE"):
            target_claim_keys.append("synchronization")
            if "baud" in ev.description.lower() or "sps" in ev.description.lower() or "timing" in ev.description.lower():
                target_claim_keys.append("symbol_rate")
        if "DEMOD" in cat or ev.independence_group.value == "DEMODULATION":
            target_claim_keys.append("demodulation")
        if "INTERLEAVER" in cat:
            target_claim_keys.append("interleaver")
        if "FEC" in cat:
            target_claim_keys.append("fec")
        if "CRC" in cat or "VALIDATION" in cat:
            target_claim_keys.append("validation")
            target_claim_keys.append("fec")
            target_claim_keys.append("payload")
        if "FRAME" in cat or "BITSTREAM" in cat:
            target_claim_keys.append("frame_structure")
        if "PAYLOAD" in cat:
            target_claim_keys.append("payload")

        for k in target_claim_keys:
            claim_id = f"claim_{k}"
            if claim_id in [f"claim_{name}" for name in field_explanations.keys()]:
                edges.append(ReasoningEdge(
                    source_id=ev_node_id,
                    target_id=claim_id,
                    edge_type=edge_type,
                    label=ev.strength.value.lower()
                ))
