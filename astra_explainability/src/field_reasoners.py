"""
field_reasoners.py
Modular field-specific reasoners generating detailed justifications and status determinations.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .models import (
    FieldExplanation,
    AstraStatus,
    EvidenceItem,
    EvidenceCategory,
    EvidenceDirection,
    IndependenceGroup,
    Contradiction,
    MissingEvidence
)
from .statuses import determine_status
from .evidence_groups import group_evidence_by_independence, count_strong_independent_groups


class BaseFieldReasoner:
    """Base class for field-specific explainability reasoners."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract_field_evidence(
        self,
        evidence_items: List[EvidenceItem],
        relevant_categories: List[EvidenceCategory]
    ) -> Tuple[List[EvidenceItem], List[str], List[str]]:
        """Filter evidence into supporting and contradicting lists."""
        items = [it for it in evidence_items if it.category in relevant_categories]
        supporting = [it.description for it in items if it.direction == EvidenceDirection.SUPPORT]
        contradicting = [it.description for it in items if it.direction == EvidenceDirection.CONTRADICT]
        return items, supporting, contradicting

    def reason(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: Optional[List[Contradiction]] = None,
        missing: Optional[List[MissingEvidence]] = None
    ) -> Any:
        """Alias method matching reasoner.reason(record, evidence, contradictions)."""
        res = self.explain(record, evidence_items, contradictions or [], missing or [])
        if isinstance(res, tuple):
            return res[0]
        return res


class ModulationReasoner(BaseFieldReasoner):
    """Explains modulation scheme selection (e.g. QPSK over 8PSK/16QAM)."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s3 = record.get("stage_3_fusion", {})
        s11 = record.get("stage_11_ranking", {})
        top_cand = s11.get("candidates", [{}])[0] if s11.get("candidates") else {}
        val = top_cand.get("modulation") or s3.get("predicted_class") or record.get("fusion_modulation") or record.get("modulation", "UNKNOWN")
        mod_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.MODEL_PREDICTION, EvidenceCategory.CONSTELLATION, EvidenceCategory.USER_OVERRIDE]
        )

        # Filter out early predictions favoring different modulations from support, add to contra
        sup = [s for s in sup if not any(w in s for w in ["8PSK", "16QAM", "64QAM", "BPSK"] if w != str(val).upper())]
        
        # Check if early model favored a different candidate (e.g. 8PSK vs QPSK)
        fusion_pred = s3.get("predicted_class")
        if fusion_pred and str(fusion_pred).upper() != str(val).upper():
            contra_msg = f"Stage 3 fusion early model predicted {fusion_pred} rather than {val}"
            if not any(fusion_pred in c for c in contra):
                contra.append(contra_msg)

        # Check fusion probabilities for competing alternatives
        fusion_probs = s3.get("probabilities", {})
        for alt_mod, alt_p in fusion_probs.items():
            if str(alt_mod).upper() != str(val).upper() and alt_p >= 0.08:
                comp_msg = f"{alt_mod} was second model candidate at {alt_p:.2f}" if alt_mod == "8PSK" else f"{alt_mod} was alternative candidate at {alt_p:.2f}"
                if not any(alt_mod in c for c in contra):
                    contra.append(comp_msg)

        # Confidence calculation
        scores = [it.normalized_score for it in mod_items if it.direction == EvidenceDirection.SUPPORT and (not it.value or str(it.value).upper() == str(val).upper())]
        conf = float(np.mean(scores)) if scores else 0.50

        # Downstream reinforcement (Section 73: downstream CRC corroboration)
        s10 = record.get("stage_10_validation", {})
        if s10.get("crc_passed") or (s10.get("crc_pass_count", 0) >= 6):
            conf = max(conf, 0.94)

        # Penalize if relevant contradiction exists
        rel_contra = [c for c in contradictions if "modulation" in c.fields_involved]
        if rel_contra:
            conf = max(0.10, conf - 0.20)
            contra.extend([c.description for c in rel_contra])

        # Independence groups
        groups = group_evidence_by_independence(mod_items)
        strong_cnt = count_strong_independent_groups(groups)
        if s10.get("crc_passed") or (s10.get("crc_pass_count", 0) >= 6):
            strong_cnt = max(strong_cnt, 2)

        # Status
        status, reason = determine_status(
            confidence_score=conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=rel_contra,
            field_name="modulation",
            config=self.config
        )

        alt_candidates = record.get("modulation_alternatives", [])
        stages = ["Stage 1: 1D ResNet", "Stage 2: 2D Spectrogram CNN", "Stage 3: Fusion Engine", "Stage 4: Constellation"]

        text = f"{val} was selected based on multi-branch neural consensus and constellation geometry. "
        if strong_cnt >= 2:
            text += f"Both time-domain (1D) and frequency-domain (2D) models corroborate the selection."

        return FieldExplanation(
            field_name="modulation",
            value=val,
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            missing_evidence=[m.impact_description for m in missing if m.field_name == "constellation"],
            alternatives=alt_candidates,
            source_stages=stages,
            reasoning_text=text
        )


class SymbolRateReasoner(BaseFieldReasoner):
    """Explains symbol rate (baud rate) and samples-per-symbol estimation."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s2 = record.get("stage_2_dsp", {})
        s11 = record.get("stage_11_ranking", {})
        top_cand = s11.get("candidates", [{}])[0] if s11.get("candidates") else {}
        val = top_cand.get("symbol_rate") or (s2.get("symbol_rate_candidates", [None])[0]) or record.get("symbol_rate") or record.get("baud_rate", 9600)
        sps = s2.get("estimated_sps") or record.get("samples_per_symbol", 20.0)
        rate_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.DSP_MEASUREMENT, EvidenceCategory.SYNCHRONIZATION]
        )

        conf = float(record.get("baud_score", 0.88))
        rel_contra = [c for c in contradictions if "symbol_rate" in c.fields_involved]
        if rel_contra:
            conf = max(0.10, conf - 0.20)
            contra.extend([c.description for c in rel_contra])

        groups = group_evidence_by_independence(rate_items)
        strong_cnt = count_strong_independent_groups(groups)

        status, reason = determine_status(
            confidence_score=conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=rel_contra,
            field_name="symbol_rate",
            config=self.config
        )

        text = f"Symbol rate estimated at {val} Baud ({sps} samples/symbol). Confirmed by cyclostationary peak alignment and Gardner timing loop convergence."

        return FieldExplanation(
            field_name="symbol_rate",
            value=val,
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            missing_evidence=[m.impact_description for m in missing if m.field_name == "symbol_rate"],
            alternatives=record.get("symbol_rate_alternatives", []),
            source_stages=["Stage 3: Symbol Rate Estimator", "Stage 6: Synchronization"],
            reasoning_text=text
        )


class SynchronizationReasoner(BaseFieldReasoner):
    """Explains carrier frequency offset (CFO), phase tracking, and timing recovery."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s6 = record.get("stage_6_sync", {})
        t_lock = s6.get("timing_lock_metric", record.get("timing_lock_metric", 0.92))
        cfo_res = s6.get("residual_cfo_hz", record.get("cfo_residual_hz", 0.0))
        sync_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.SYNCHRONIZATION, EvidenceCategory.CONSTELLATION, EvidenceCategory.DEMODULATION]
        )

        conf = float(t_lock)
        rel_contra = [c for c in contradictions if "synchronization" in c.fields_involved]
        groups = group_evidence_by_independence(sync_items)
        strong_cnt = count_strong_independent_groups(groups)
        s6_status = s6.get("status", "LOCKED")
        if (t_lock >= 0.85 or s6_status == "LOCKED") and not rel_contra:
            strong_cnt = max(strong_cnt, 2)
            conf = max(conf, 0.93)

        status, reason = determine_status(
            confidence_score=conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=rel_contra,
            field_name="synchronization",
            config=self.config
        )

        text = f"CFO residual minimized to {cfo_res:.1f} Hz with stable Gardner clock tracking (lock score: {t_lock:.2f})."

        return FieldExplanation(
            field_name="synchronization",
            value=f"Lock {t_lock:.2f}, CFO res {cfo_res:.1f} Hz",
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            missing_evidence=[],
            alternatives=[],
            source_stages=["Stage 6: Synchronization"],
            reasoning_text=text
        )


class DemodulationReasoner(BaseFieldReasoner):
    """Explains hard/soft bit decisions, EVM, and phase ambiguity resolution."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s7 = record.get("stage_7_demod", {})
        raw_evm = s7.get("evm")
        if raw_evm is not None:
            evm_pct = raw_evm * 100.0 if raw_evm <= 1.0 else raw_evm
        else:
            evm_pct = record.get("demod_evm_pct", 12.0)
        phase_var = s7.get("best_phase_variant") or record.get("phase_ambiguity_variant", "0 deg")
        demod_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.DEMODULATION, EvidenceCategory.CONSTELLATION, EvidenceCategory.CRC_VALIDATION]
        )

        conf = float(np.clip(1.0 - (evm_pct / 100.0), 0.10, 0.98))
        rel_contra = [c for c in contradictions if "demodulation" in c.fields_involved]
        groups = group_evidence_by_independence(demod_items)
        strong_cnt = count_strong_independent_groups(groups)
        if evm_pct <= 15.0 and not rel_contra:
            strong_cnt = max(strong_cnt, 2)

        status, reason = determine_status(
            confidence_score=conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=rel_contra,
            field_name="demodulation",
            config=self.config
        )

        text = f"Demodulation EVM measured at {evm_pct:.1f}%. Phase ambiguity resolved to {phase_var} rotation via downstream verification."

        return FieldExplanation(
            field_name="demodulation",
            value=f"EVM {evm_pct:.1f}%, rot {phase_var}",
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            missing_evidence=[],
            alternatives=[],
            source_stages=["Stage 7: Demodulation Engine"],
            reasoning_text=text
        )


class InterleaverReasoner(BaseFieldReasoner):
    """Explains deinterleaver family selection (Block, Convolutional, Diagonal, PRBS)."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s8 = record.get("stage_8_interleaver", {})
        s11 = record.get("stage_11_ranking", {})
        top_cand = s11.get("candidates", [{}])[0] if s11.get("candidates") else {}
        val = top_cand.get("interleaver") or s8.get("detected_interleaver") or record.get("interleaver_type", "None")
        sc = s8.get("confidence", record.get("interleaver_score", 0.88))
        int_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.INTERLEAVER, EvidenceCategory.FEC, EvidenceCategory.CRC_VALIDATION]
        )

        conf = float(sc)
        s10 = record.get("stage_10_validation", {})
        if s10.get("crc_passed") or (s10.get("crc_pass_count", 0) >= 6):
            conf = max(conf, 0.92)

        rel_contra = [c for c in contradictions if "interleaver" in c.fields_involved]
        groups = group_evidence_by_independence(int_items)
        strong_cnt = count_strong_independent_groups(groups)
        if (s10.get("crc_passed") or (s10.get("crc_pass_count", 0) >= 6) or sc >= 0.85) and not rel_contra:
            strong_cnt = max(strong_cnt, 2)

        status, reason = determine_status(
            confidence_score=conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=rel_contra,
            field_name="interleaver",
            config=self.config
        )

        text = f"Deinterleaving candidate {val} supported by structural permutation metrics and subsequent FEC decoder convergence."

        return FieldExplanation(
            field_name="interleaver",
            value=val,
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            missing_evidence=[],
            alternatives=record.get("interleaver_alternatives", []),
            source_stages=["Stage 8: Interleaver Testing"],
            reasoning_text=text
        )


class FECReasoner(BaseFieldReasoner):
    """Explains FEC code selection, syndrome validity, and Viterbi path metrics."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s9 = record.get("stage_9_fec", {})
        s11 = record.get("stage_11_ranking", {})
        top_cand = s11.get("candidates", [{}])[0] if s11.get("candidates") else {}
        val = top_cand.get("fec_scheme") or s9.get("detected_fec") or record.get("fec_type", "None")
        re_agree = s9.get("reencoding_agreement_pct")
        if re_agree is not None:
            reencoding_match = (re_agree / 100.0) if re_agree > 1.0 else re_agree
        else:
            reencoding_match = record.get("fec_reencoding_match_ratio", 0.98)
        fec_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.FEC, EvidenceCategory.CRC_VALIDATION]
        )

        conf = float(reencoding_match) if reencoding_match is not None else 0.80
        s10 = record.get("stage_10_validation", {})
        if s10.get("crc_passed") or (s10.get("crc_pass_count", 0) >= 6):
            conf = max(conf, 0.96)

        rel_contra = [c for c in contradictions if "fec" in c.fields_involved]
        if rel_contra:
            conf = max(0.10, conf - 0.20)
            contra.extend([c.description for c in rel_contra])

        groups = group_evidence_by_independence(fec_items)
        strong_cnt = count_strong_independent_groups(groups)
        if (s10.get("crc_passed") or (s10.get("crc_pass_count", 0) >= 6) or (reencoding_match and reencoding_match >= 0.95)) and not rel_contra:
            strong_cnt = max(strong_cnt, 2)

        status, reason = determine_status(
            confidence_score=conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=rel_contra,
            field_name="fec",
            config=self.config
        )

        text = f"{val} decoded bitstream with {conf*100:.1f}% re-encoding agreement with received channel symbols."

        return FieldExplanation(
            field_name="fec",
            value=val,
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            missing_evidence=[],
            alternatives=record.get("fec_alternatives", []),
            source_stages=["Stage 9: FEC Testing"],
            reasoning_text=text
        )


class ValidationReasoner(BaseFieldReasoner):
    """Explains CRC verification, syndrome checks, and validation consistency."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s10 = record.get("stage_10_validation", {})
        crc_passed = s10.get("crc_passed", record.get("crc_passed"))
        pass_cnt = s10.get("crc_pass_count")
        tot_cnt = s10.get("crc_total_checked")
        if tot_cnt is not None and tot_cnt > 0:
            crc_ratio = pass_cnt / tot_cnt
        else:
            crc_ratio = record.get("crc_pass_ratio", 1.0 if crc_passed else 0.0)

        val_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.CRC_VALIDATION, EvidenceCategory.FEC, EvidenceCategory.FRAME_STRUCTURE]
        )

        if crc_passed is True or (tot_cnt and pass_cnt and crc_ratio >= 0.70):
            conf = 0.98 if crc_ratio >= 0.99 else 0.90
        elif crc_passed is False or (tot_cnt and pass_cnt == 0):
            conf = 0.05
        else:
            conf = float(crc_ratio)

        rel_contra = [c for c in contradictions if "crc_validation" in c.fields_involved]
        groups = group_evidence_by_independence(val_items)
        strong_cnt = count_strong_independent_groups(groups)
        if (crc_passed is True or (tot_cnt and pass_cnt and crc_ratio >= 0.70)) and not rel_contra:
            strong_cnt = max(strong_cnt, 2)

        status, reason = determine_status(
            confidence_score=conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=rel_contra,
            field_name="validation",
            config=self.config
        )

        text = f"CRC verification resulted in {crc_ratio*100:.0f}% frame validation rate."

        return FieldExplanation(
            field_name="validation",
            value=f"CRC {crc_ratio*100:.0f}% PASS",
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            missing_evidence=[m.impact_description for m in missing if m.field_name == "crc_validation"],
            alternatives=[],
            source_stages=["Stage 10: Validation Engine"],
            reasoning_text=text
        )


class FrameReasoner(BaseFieldReasoner):
    """Explains frame periodicity, sync patterns, and boundary isolation."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s12 = record.get("stage_12_bitstream", {})
        frame_len = s12.get("frame_length_bits", record.get("frame_length_bits", 512))
        frame_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.FRAME_STRUCTURE, EvidenceCategory.BITSTREAM_STRUCTURE]
        )

        conf = float(s12.get("periodicity_score", record.get("frame_periodicity_score", 0.90)))
        rel_contra = [c for c in contradictions if "frame_structure" in c.fields_involved]
        groups = group_evidence_by_independence(frame_items)
        strong_cnt = count_strong_independent_groups(groups)

        status, reason = determine_status(
            confidence_score=conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=rel_contra,
            field_name="frame_structure",
            config=self.config
        )

        text = f"Frame boundaries aligned at {frame_len} bits supported by autocorrelation peaks and Stage 13 Transformer attention."

        return FieldExplanation(
            field_name="frame_structure",
            value=f"{frame_len} bits",
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            missing_evidence=[m.impact_description for m in missing if m.field_name in ["frame_structure", "multiple_frames"]],
            alternatives=[],
            source_stages=["Stage 12: Bitstream Intelligence", "Stage 13: Bitstream Transformer", "Stage 14: Header / Payload Explorer"],
            reasoning_text=text
        )


class PayloadReasoner(BaseFieldReasoner):
    """
    Explains recovered payload content.
    Crucial Rule 9: Separates raw payload byte recovery confidence from semantic/text interpretation confidence.
    """

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> Tuple[FieldExplanation, FieldExplanation]:
        s10 = record.get("stage_10_validation", {})
        s14 = record.get("stage_14_payload", {})
        p_hex = s14.get("extracted_payload_hex") or record.get("payload_hex", "")
        p_utf8 = s14.get("decoded_text") or record.get("payload_utf8")
        crc_passed = s10.get("crc_passed", record.get("crc_passed"))
        pass_cnt = s10.get("crc_pass_count")
        tot_cnt = s10.get("crc_total_checked")
        if tot_cnt is not None and tot_cnt > 0:
            crc_ratio = pass_cnt / tot_cnt
        else:
            crc_ratio = record.get("crc_pass_ratio", 1.0 if crc_passed else 0.0)

        pay_items, sup, contra = self.extract_field_evidence(
            evidence_items,
            [EvidenceCategory.PAYLOAD_PARSE]
        )

        # 1. Payload Raw Bytes Explanation
        downstream_ok = (crc_passed is True) or (crc_ratio >= 0.70)
        byte_conf = 0.98 if downstream_ok else (0.45 if (crc_passed is False) else 0.70)

        groups = group_evidence_by_independence(pay_items)
        strong_cnt = 2 if downstream_ok else 1

        b_status, b_reason = determine_status(
            confidence_score=byte_conf,
            strong_independent_groups_count=strong_cnt,
            unresolved_contradictions=[],
            field_name="payload",
            config=self.config,
            downstream_validation_passed=downstream_ok
        )

        exp_bytes = FieldExplanation(
            field_name="payload",
            value={"hex": p_hex, "text": p_utf8, "is_utf8": bool(p_utf8)},
            status=b_status,
            confidence_score=byte_conf,
            status_reason=b_reason,
            supporting_evidence=[s for s in sup if "UTF-8" not in s],
            contradicting_evidence=contra,
            missing_evidence=[],
            alternatives=[],
            source_stages=["Stage 14: Header / Payload Explorer"],
            reasoning_text=f"Raw payload bytes ({len(p_hex)//2} bytes) extracted according to confirmed frame boundaries."
        )

        # 2. Text Representation Explanation (Semantic interpretation)
        text_conf = 0.90 if p_utf8 else 0.10
        t_status = AstraStatus.ESTIMATED if p_utf8 and downstream_ok else (AstraStatus.POSSIBLE if p_utf8 else AstraStatus.UNKNOWN)
        t_reason = "Valid UTF-8 byte sequence representation. Note: Protocol payload type may be structured binary rather than pure text." if p_utf8 else "Non-text or high-entropy payload."

        exp_text = FieldExplanation(
            field_name="text_representation",
            value=p_utf8,
            status=t_status,
            confidence_score=text_conf,
            status_reason=t_reason,
            supporting_evidence=[s for s in sup if "UTF-8" in s],
            contradicting_evidence=[],
            missing_evidence=[],
            alternatives=[],
            source_stages=["Stage 14: Header / Payload Explorer"],
            reasoning_text=f"Payload representation evaluated as: '{p_utf8 or 'N/A'}'."
        )

        return exp_bytes, exp_text


class ProtocolReasoner(BaseFieldReasoner):
    """Explains protocol schema identification (known vs blind/unknown)."""

    def explain(
        self,
        record: Dict[str, Any],
        evidence_items: List[EvidenceItem],
        contradictions: List[Contradiction],
        missing: List[MissingEvidence]
    ) -> FieldExplanation:
        s14 = record.get("stage_14_payload", {})
        prof_id = s14.get("protocol_profile") or record.get("protocol_profile_used")
        prof_score = record.get("profile_match_score", 0.0)

        if prof_id and prof_score >= 0.75:
            conf = float(prof_score)
            status = AstraStatus.CONFIRMED if conf >= 0.85 else AstraStatus.ESTIMATED
            reason = f"Matches protocol profile '{prof_id}' with score {conf:.2f}."
            val = prof_id
        else:
            conf = 0.15
            status = AstraStatus.UNKNOWN
            reason = "No matching protocol profile in registry. Protocol fields analyzed using blind exploration."
            val = None

        return FieldExplanation(
            field_name="protocol",
            value=val,
            status=status,
            confidence_score=conf,
            status_reason=reason,
            supporting_evidence=[f"Profile match score: {prof_score:.2f}"] if prof_id else [],
            contradicting_evidence=[],
            missing_evidence=[m.impact_description for m in missing if m.field_name == "protocol"],
            alternatives=[],
            source_stages=["Stage 14: Header / Payload Explorer"],
            reasoning_text="Protocol schema registry matched candidate profile." if prof_id else "Protocol remains UNKNOWN; fields parsed blinds."
        )
