"""
ASTRA Stage 15: Summary & GUI Card Generator.

Generates:
1. Human-readable concise summary
2. Detailed engineering/technical summary
3. Compact GUI cards, badges, timeline, and warnings
"""

from typing import Dict, List, Any
from .models import (
    FieldExplanation, CandidateExplanation, Contradiction,
    MissingEvidence, GuiSummary, GuiCard, AstraStatus
)


class SummaryGenerator:
    """Generates multi-level summaries and GUI presentation artifacts."""

    def generate_human_summary(
        self,
        overall_status: AstraStatus,
        overall_confidence: float,
        fields: Dict[str, FieldExplanation],
        contradictions: List[Contradiction]
    ) -> str:
        """
        Creates a clean, human-readable narrative.
        Example:
        ASTRA recovered a QPSK signal at approximately 9600 baud. Synchronization achieved stable
        timing and carrier lock. The Block 16x16 + convolutional K=7 rate-1/2 path produced
        repeated valid CRCs and consistent 512-bit frames. The recovered 8-byte payload is
        '68 69 20 68 65 6c 6c 6f', which is valid UTF-8 for 'hi hello'.
        """
        mod_exp = fields.get("modulation")
        sr_exp = fields.get("symbol_rate")
        sync_exp = fields.get("synchronization")
        int_exp = fields.get("interleaver")
        fec_exp = fields.get("fec")
        val_exp = fields.get("validation")
        frm_exp = fields.get("frame_structure")
        pay_exp = fields.get("payload")
        txt_exp = fields.get("text_representation")
        proto_exp = fields.get("protocol")

        sentences = []

        # Mod & Baud
        mod_val = mod_exp.value if mod_exp else "Unknown"
        sr_val = f"{sr_exp.value} baud" if sr_exp and sr_exp.value else "unknown baud"
        sentences.append(f"ASTRA recovered a {mod_val} signal operating at approximately {sr_val}.")

        # Sync
        if sync_exp and sync_exp.status in (AstraStatus.CONFIRMED, AstraStatus.ESTIMATED):
            sentences.append("Synchronization achieved stable timing and carrier tracking.")

        # FEC / Interleaver / Validation
        chain_parts = []
        if int_exp and int_exp.value not in (None, "None", "None / Unknown"):
            chain_parts.append(str(int_exp.value))
        if fec_exp and fec_exp.value not in (None, "None", "None / Unknown"):
            chain_parts.append(str(fec_exp.value))
        chain_str = " + ".join(chain_parts) if chain_parts else "unencoded / raw"

        if val_exp and val_exp.status == AstraStatus.CONFIRMED:
            sentences.append(f"The {chain_str} processing path produced repeated valid frame CRC checks.")
        elif fec_exp and fec_exp.status in (AstraStatus.CONFIRMED, AstraStatus.ESTIMATED):
            sentences.append(f"The {chain_str} path achieved decoder convergence and metric validity.")

        # Frame & Payload
        if frm_exp and frm_exp.value:
            flen = frm_exp.value.get("estimated_frame_len_bits") if isinstance(frm_exp.value, dict) else frm_exp.value
            if flen:
                sentences.append(f"Frame periodicity indicates regular {flen}-bit frames.")

        if pay_exp and pay_exp.value:
            p_val = pay_exp.value
            hex_val = p_val.get("hex", "") if isinstance(p_val, dict) else str(p_val)
            byte_len = p_val.get("length_bytes", len(hex_val)//2) if isinstance(p_val, dict) else ""
            len_str = f"{byte_len}-byte " if byte_len else ""

            if txt_exp and txt_exp.value:
                sentences.append(
                    f"Recovered {len_str}payload bytes (`{hex_val}`), representing valid text: '{txt_exp.value}'."
                )
            else:
                sentences.append(f"Recovered {len_str}payload bytes with hex content `{hex_val}`.")

        # Protocol
        if proto_exp and proto_exp.status == AstraStatus.UNKNOWN:
            sentences.append("Protocol profile remained unrecognized (blind exploratory recovery).")
        elif proto_exp and proto_exp.status in (AstraStatus.CONFIRMED, AstraStatus.ESTIMATED):
            sentences.append(f"Protocol matched signature profile: {proto_exp.value}.")

        # Critical contradictions / warnings
        high_contras = [c for c in contradictions if c.severity.value in ("HIGH", "CRITICAL")]
        if high_contras:
            sentences.append(f"Caution: {len(high_contras)} critical contradiction(s) detected during analysis.")

        return " ".join(sentences)

    def generate_technical_summary(
        self,
        overall_status: AstraStatus,
        overall_confidence: float,
        fields: Dict[str, FieldExplanation],
        candidates: List[CandidateExplanation],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> str:
        """Generates detailed engineering summary with metrics, margins, and status breakdowns."""
        lines = []
        lines.append("=== ASTRA STAGE 15: TECHNICAL EXPLAINABILITY REPORT ===")
        lines.append(f"Overall Status: {overall_status.value} | ASTRA Reasoning Confidence: {overall_confidence:.4f}")
        lines.append("")

        lines.append("--- FIELD-SPECIFIC STATUS & CONFIDENCE BREAKDOWN ---")
        for f_name, f_exp in fields.items():
            lines.append(
                f"[{f_exp.status.value:<9}] {f_name:<20}: {str(f_exp.value):<28} "
                f"Conf: {f_exp.confidence_score:.3f} | Stages: {f_exp.source_stages}"
            )
            if f_exp.status_reason:
                lines.append(f"  Reason: {f_exp.status_reason}")
            if f_exp.supporting_evidence:
                lines.append(f"  Support ({len(f_exp.supporting_evidence)} items): {f_exp.supporting_evidence[0]}")
            if f_exp.contradicting_evidence:
                lines.append(f"  Contradict ({len(f_exp.contradicting_evidence)} items): {f_exp.contradicting_evidence[0]}")
        lines.append("")

        if candidates:
            lines.append("--- CANDIDATE HYPOTHESIS COMPARISON ---")
            for c in candidates:
                winner_tag = " [BEST CANDIDATE]" if c.is_best else f" [RANK #{c.rank}]"
                lines.append(f"{c.pipeline_id}{winner_tag} - Score: {c.score:.4f} (Margin to next: {c.margin_to_next:.4f})")
                lines.append(f"  Configuration: Mod={c.modulation}, Baud={c.symbol_rate}, FEC={c.fec_scheme}, Interleaver={c.interleaver}")
                if not c.is_best and c.why_lost:
                    lines.append(f"  Why lost: {c.why_lost}")
            lines.append("")

        lines.append(f"--- CONTRADICTIONS DETECTED ({len(contradictions)}) ---")
        if contradictions:
            for c in contradictions:
                lines.append(f"[{c.severity.value}] {c.contradiction_id}: {c.description} (Fields: {c.fields_involved})")
        else:
            lines.append("None. All multi-stage signals are mutually consistent.")
        lines.append("")

        lines.append(f"--- MISSING EVIDENCE AUDIT ({len(missing)}) ---")
        if missing:
            for m in missing:
                lines.append(f"Missing '{m.evidence_type}': {m.impact_description} (Confidence penalty: -{m.penalty:.3f})")
        else:
            lines.append("Complete. Full Stage 1-14 evidence suite available.")

        return "\n".join(lines)

    def generate_gui_summary(
        self,
        overall_status: AstraStatus,
        overall_confidence: float,
        fields: Dict[str, FieldExplanation],
        candidates: List[CandidateExplanation],
        contradictions: List[Contradiction]
    ) -> GuiSummary:
        """Creates compact cards, badges, warnings, and timeline for Stage 16 GUI."""
        cards: List[GuiCard] = []

        card_keys = [
            ("modulation", "Modulation", "Modulation Family"),
            ("symbol_rate", "Baud Rate", "Symbol Rate"),
            ("synchronization", "Sync Quality", "Timing & CFO"),
            ("fec", "FEC Scheme", "Forward Error Correction"),
            ("validation", "CRC / Parity", "Downstream Integrity"),
            ("payload", "Payload", "Decoded Content"),
            ("protocol", "Protocol", "Protocol Profile")
        ]

        for k, title, subtitle in card_keys:
            if k in fields:
                f = fields[k]
                disp_val = f.value
                if isinstance(disp_val, dict):
                    disp_val = disp_val.get("hex") or disp_val.get("profile_id") or str(disp_val)
                elif disp_val is None:
                    disp_val = "Unknown / Unmatched"

                cards.append(GuiCard(
                    field_name=k,
                    title=title,
                    value_display=str(disp_val),
                    status=f.status,
                    confidence_pct=int(round(f.confidence_score * 100)),
                    confidence_display=f"{int(round(f.confidence_score * 100))}%",
                    badge_color=self._get_badge_color(f.status),
                    subtitle=subtitle
                ))

        # Overall badge color
        overall_badge = self._get_badge_color(overall_status)

        # Warnings generator
        warnings = self._generate_warnings(fields, candidates, contradictions)

        # Timeline generator
        timeline = self._generate_timeline(fields)

        return GuiSummary(
            overall_status=overall_status,
            overall_confidence_badge=f"ASTRA Confidence: {int(round(overall_confidence * 100))}%",
            cards=cards,
            warnings=warnings,
            timeline_items=timeline
        )

    def _get_badge_color(self, status: AstraStatus) -> str:
        if status == AstraStatus.CONFIRMED:
            return "green"
        elif status == AstraStatus.ESTIMATED:
            return "blue"
        elif status == AstraStatus.POSSIBLE:
            return "amber"
        return "gray"

    def _generate_warnings(
        self,
        fields: Dict[str, FieldExplanation],
        candidates: List[CandidateExplanation],
        contradictions: List[Contradiction]
    ) -> List[str]:
        warnings = []

        # Contradictions
        for c in contradictions:
            if c.severity.value in ("HIGH", "CRITICAL"):
                warnings.append(f"CONTRADICTION: {c.description}")

        # Low confidence fields
        for fname, fexp in fields.items():
            if fexp.status == AstraStatus.UNKNOWN and fname in ("validation", "payload"):
                warnings.append(f"VALIDATION_UNCERTAIN: {fname} status is UNKNOWN.")

        # Candidate competition
        if len(candidates) >= 2:
            top1 = candidates[0]
            top2 = candidates[1]
            if (top1.score - top2.score) < 0.05:
                warnings.append(
                    f"CLOSE_PIPELINE_COMPETITION: Top 2 candidates nearly tied ({top1.pipeline_id} vs {top2.pipeline_id})."
                )

        # Missing CRC
        val = fields.get("validation")
        if val and val.value in ("No CRC Checked", None, "UNKNOWN"):
            warnings.append("CRC_UNAVAILABLE: Validation conducted without deterministic CRC.")

        # Protocol unknown
        proto = fields.get("protocol")
        if proto and proto.status == AstraStatus.UNKNOWN:
            warnings.append("PROTOCOL_UNKNOWN: Bitstream payload could not be mapped to any known protocol signature.")

        return warnings

    def _generate_timeline(self, fields: Dict[str, FieldExplanation]) -> List[str]:
        timeline = []
        timeline_order = [
            ("modulation", "Stage 3-4", "Modulation hypothesis supported"),
            ("symbol_rate", "Stage 5", "Symbol rate and SPS candidate selected"),
            ("synchronization", "Stage 6", "Carrier and timing synchronization completed"),
            ("demodulation", "Stage 7", "Demodulation constellation mapped and soft LLRs extracted"),
            ("interleaver", "Stage 8", "Interleaver permutation resolved"),
            ("fec", "Stage 9", "FEC decoding converged"),
            ("validation", "Stage 10", "Frame integrity validation executed"),
            ("frame_structure", "Stage 12-13", "Bitstream frame boundaries detected"),
            ("payload", "Stage 14", "Payload extracted and segmented")
        ]

        for key, stage_str, default_msg in timeline_order:
            if key in fields:
                f = fields[key]
                msg = f"{stage_str}: {key.replace('_', ' ').title()} = {f.value} ({f.status.value})"
                timeline.append(msg)

        timeline.append("Stage 15: Explainability & Confidence Reasoning Engine synthesized all evidence")
        return timeline
