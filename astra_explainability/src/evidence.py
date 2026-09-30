"""
evidence.py
Extraction, normalization, and categorization of atomic evidence items from Stages 1–14.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np

from .models import (
    EvidenceItem,
    EvidenceCategory,
    EvidenceStrength,
    EvidenceDirection,
    IndependenceGroup
)


def create_evidence_item(
    evidence_id: str,
    stage: str,
    category: EvidenceCategory,
    description: str,
    value: Any = None,
    normalized_score: float = 1.0,
    strength: EvidenceStrength = EvidenceStrength.MODERATE,
    direction: EvidenceDirection = EvidenceDirection.SUPPORT,
    independence_group: IndependenceGroup = IndependenceGroup.DSP_SIGNAL_STRUCTURE,
    source_reference: Optional[str] = None
) -> EvidenceItem:
    """Factory helper to instantiate a verified EvidenceItem."""
    score = float(np.clip(normalized_score, 0.0, 1.0))
    return EvidenceItem(
        evidence_id=evidence_id,
        stage=stage,
        category=category,
        description=description,
        value=value,
        normalized_score=score,
        strength=strength,
        direction=direction,
        independence_group=independence_group,
        source_reference=source_reference
    )


class EvidenceExtractor:
    """
    Unified Evidence Extractor across all upstream ASTRA stages (1-14).
    Supports both flat records and nested stage_X dictionary formats.
    """

    def extract_all(self, record: Dict[str, Any]) -> List[EvidenceItem]:
        return extract_evidence_from_analysis_record(record)


def extract_evidence_from_analysis_record(record: Dict[str, Any]) -> List[EvidenceItem]:
    """
    Extract, normalize, and categorize evidence across all upstream ASTRA stages.
    Handles both nested structures (e.g. stage_3_fusion) and flattened records.
    """
    items: List[EvidenceItem] = []

    # Helper sub-dicts
    s1 = record.get("stage_1_signal", {})
    s2 = record.get("stage_2_dsp", {})
    s3 = record.get("stage_3_fusion", {})
    s4 = record.get("stage_4_constellation", {})
    s5 = record.get("stage_5_candidates", {})
    s6 = record.get("stage_6_sync", {})
    s7 = record.get("stage_7_demod", {})
    s8 = record.get("stage_8_interleaver", {})
    s9 = record.get("stage_9_fec", {})
    s10 = record.get("stage_10_validation", {})
    s11 = record.get("stage_11_ranking", {})
    s12 = record.get("stage_12_bitstream", {})
    s13 = record.get("stage_13_transformer", {})
    s14 = record.get("stage_14_payload", {})

    # 1. Modulation Evidence (Stages 1-4)
    # 1D ResNet
    resnet1d = s3.get("1d_resnet", {})
    resnet_prob = record.get("resnet1d_prob")
    resnet_mod = record.get("resnet1d_modulation")
    if not resnet_prob and resnet1d:
        # Get top class and prob
        sorted_1d = sorted(resnet1d.items(), key=lambda x: x[1], reverse=True)
        if sorted_1d:
            resnet_mod, resnet_prob = sorted_1d[0]

    if resnet_prob is not None:
        p = float(resnet_prob)
        st = EvidenceStrength.STRONG if p >= 0.80 else (EvidenceStrength.MODERATE if p >= 0.50 else EvidenceStrength.WEAK)
        items.append(create_evidence_item(
            evidence_id="ev_mod_1d_resnet",
            stage="Stage 1: 1D ResNet",
            category=EvidenceCategory.MODEL_PREDICTION,
            description=f"1D ResNet raw-IQ prediction: {resnet_mod} (prob {p:.2f})",
            value=resnet_mod,
            normalized_score=p,
            strength=st,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.MODULATION_MODELS,
            source_reference="best_model_resnet1d.pt"
        ))

    # 2D Spectrogram CNN
    cnn2d = s3.get("2d_cnn", {})
    cnn2d_prob = record.get("cnn2d_prob")
    cnn2d_mod = record.get("cnn2d_modulation")
    if not cnn2d_prob and cnn2d:
        sorted_2d = sorted(cnn2d.items(), key=lambda x: x[1], reverse=True)
        if sorted_2d:
            cnn2d_mod, cnn2d_prob = sorted_2d[0]

    if cnn2d_prob is not None:
        p = float(cnn2d_prob)
        st = EvidenceStrength.STRONG if p >= 0.80 else (EvidenceStrength.MODERATE if p >= 0.50 else EvidenceStrength.WEAK)
        items.append(create_evidence_item(
            evidence_id="ev_mod_2d_cnn",
            stage="Stage 2: 2D Spectrogram CNN",
            category=EvidenceCategory.MODEL_PREDICTION,
            description=f"2D Spectrogram CNN prediction: {cnn2d_mod} (prob {p:.2f})",
            value=cnn2d_mod,
            normalized_score=p,
            strength=st,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.MODULATION_MODELS,
            source_reference="best_model_2dcnn.pt"
        ))

    # Fusion Engine
    fusion_mod = s3.get("predicted_class") or record.get("fusion_modulation")
    fusion_probs = s3.get("probabilities", {})
    fusion_prob = fusion_probs.get(fusion_mod) if fusion_mod else record.get("fusion_prob")
    if fusion_prob is not None:
        p = float(fusion_prob)
        items.append(create_evidence_item(
            evidence_id="ev_mod_fusion",
            stage="Stage 3: Fusion Engine",
            category=EvidenceCategory.MODEL_PREDICTION,
            description=f"Fused 1D/2D consensus: {fusion_mod} (prob {p:.2f})",
            value=fusion_mod,
            normalized_score=p,
            strength=EvidenceStrength.STRONG if p >= 0.80 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.MODULATION_MODELS,
            source_reference="astra_fusion"
        ))

    # Second model candidate (if competing)
    if fusion_probs:
        sorted_probs = sorted(fusion_probs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) > 1:
            second_mod, second_p = sorted_probs[1]
            if second_p >= 0.08:
                items.append(create_evidence_item(
                    evidence_id="ev_mod_fusion_competing",
                    stage="Stage 3: Fusion Engine",
                    category=EvidenceCategory.MODEL_PREDICTION,
                    description=f"{second_mod} was second model candidate at {second_p:.2f}",
                    value=second_mod,
                    normalized_score=second_p,
                    strength=EvidenceStrength.WEAK,
                    direction=EvidenceDirection.CONTRADICT,
                    independence_group=IndependenceGroup.MODULATION_MODELS,
                    source_reference="astra_fusion"
                ))

    # Constellation Evidence
    cluster_count = s4.get("num_clusters") or record.get("constellation_clusters")
    constellation_match = s4.get("best_match") or record.get("constellation_best_match")
    if cluster_count is not None:
        items.append(create_evidence_item(
            evidence_id="ev_constellation_clusters",
            stage="Stage 4: Constellation Engine",
            category=EvidenceCategory.CONSTELLATION,
            description=f"Post-sync constellation geometry isolated {cluster_count} clusters (match: {constellation_match or 'compact'})",
            value=cluster_count,
            normalized_score=0.92 if cluster_count in [2, 4, 8, 16, 64] else 0.5,
            strength=EvidenceStrength.STRONG if cluster_count in [2, 4, 8, 16, 64] else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.CONSTELLATION,
            source_reference="astra_constellation"
        ))

    # 2. Symbol Rate Evidence (Stage 2 / 3 / 5)
    baud_candidates = s2.get("symbol_rate_candidates", [])
    baud_rate = baud_candidates[0] if baud_candidates else (record.get("symbol_rate") or record.get("baud_rate"))
    sps = s2.get("estimated_sps") or record.get("samples_per_symbol", 20.0)
    baud_score = record.get("baud_score", 0.90)
    if baud_rate is not None:
        items.append(create_evidence_item(
            evidence_id="ev_symbol_rate",
            stage="Stage 3: Symbol Rate Estimator",
            category=EvidenceCategory.DSP_MEASUREMENT,
            description=f"Symbol rate candidate: {baud_rate} Baud (score {baud_score:.2f}, {sps} SPS)",
            value=baud_rate,
            normalized_score=float(baud_score),
            strength=EvidenceStrength.STRONG if baud_score >= 0.85 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.DSP_SIGNAL_STRUCTURE,
            source_reference="astra_symbol_rate"
        ))

    # 3. Synchronization Evidence (Stage 6)
    cfo_hz = s6.get("cfo_hz", record.get("cfo_hz", 0.0))
    residual_cfo = s6.get("residual_cfo_hz", record.get("cfo_residual_hz", 0.0))
    timing_lock = s6.get("timing_lock_metric", record.get("timing_lock_metric", 0.92))
    carrier_lock = s6.get("carrier_lock_metric", record.get("carrier_lock_metric", 0.90))
    if timing_lock is not None:
        t_score = float(timing_lock)
        items.append(create_evidence_item(
            evidence_id="ev_timing_lock",
            stage="Stage 6: Synchronization",
            category=EvidenceCategory.SYNCHRONIZATION,
            description=f"Timing recovery lock stability: {t_score:.2f} (CFO: {cfo_hz:.1f} Hz, residual: {residual_cfo:.1f} Hz)",
            value=t_score,
            normalized_score=t_score,
            strength=EvidenceStrength.STRONG if t_score >= 0.85 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.SYNC,
            source_reference="astra_synchronization"
        ))

    # 4. Demodulation Evidence (Stage 7)
    evm = s7.get("evm", record.get("demod_evm_pct"))
    # Normalize EVM to fractional if given as percentage
    if evm is not None and evm > 1.0:
        evm = evm / 100.0
    phase_variant = s7.get("best_phase_variant") or record.get("phase_ambiguity_variant", "rot0")
    if evm is not None:
        evm_score = float(np.clip(1.0 - evm, 0.0, 1.0))
        items.append(create_evidence_item(
            evidence_id="ev_demod_evm",
            stage="Stage 7: Demodulation",
            category=EvidenceCategory.DEMODULATION,
            description=f"Demodulation EVM: {evm*100:.1f}% (best phase variant: {phase_variant})",
            value=evm,
            normalized_score=evm_score,
            strength=EvidenceStrength.STRONG if evm < 0.15 else (EvidenceStrength.MODERATE if evm < 0.30 else EvidenceStrength.WEAK),
            direction=EvidenceDirection.SUPPORT if evm < 0.35 else EvidenceDirection.CONTRADICT,
            independence_group=IndependenceGroup.DEMODULATION,
            source_reference="astra_demodulation"
        ))

    # 5. Interleaver Evidence (Stage 8)
    interleaver_type = s8.get("detected_interleaver") or record.get("interleaver_type")
    interleaver_score = s8.get("confidence", record.get("interleaver_score", 0.85))
    if interleaver_type is not None:
        items.append(create_evidence_item(
            evidence_id="ev_interleaver",
            stage="Stage 8: Interleaver",
            category=EvidenceCategory.INTERLEAVER,
            description=f"Deinterleaver candidate: {interleaver_type} (score {interleaver_score:.2f})",
            value=interleaver_type,
            normalized_score=float(interleaver_score),
            strength=EvidenceStrength.STRONG if interleaver_score >= 0.85 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.FEC_INTERNAL,
            source_reference="astra_interleaver"
        ))

    # 6. FEC Evidence (Stage 9)
    fec_type = s9.get("detected_fec") or record.get("fec_type")
    reencoding_match = s9.get("reencoding_agreement_pct", record.get("fec_reencoding_match_ratio", 98.0))
    if reencoding_match > 1.0:
        reencoding_match = reencoding_match / 100.0
    syndrome_zero = s9.get("syndrome_valid", record.get("syndrome_all_zero", True))
    if fec_type is not None:
        re_score = float(reencoding_match)
        items.append(create_evidence_item(
            evidence_id="ev_fec_decoder",
            stage="Stage 9: FEC Testing",
            category=EvidenceCategory.FEC,
            description=f"FEC decoded successfully: {fec_type} (re-encoding match: {re_score*100:.1f}%)",
            value=fec_type,
            normalized_score=re_score,
            strength=EvidenceStrength.STRONG if re_score >= 0.95 and syndrome_zero else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.FEC_INTERNAL,
            source_reference="astra_fec"
        ))

    # 7. Validation & CRC Evidence (Stage 10)
    crc_passed = s10.get("crc_passed", record.get("crc_passed"))
    pass_cnt = s10.get("crc_pass_count")
    total_cnt = s10.get("crc_total_checked")
    crc_type = s10.get("crc_type")

    if total_cnt is not None and total_cnt > 0:
        ratio = float(pass_cnt / total_cnt)
        is_pass = ratio >= 0.80
        items.append(create_evidence_item(
            evidence_id="ev_crc_validation",
            stage="Stage 10: Validation Engine",
            category=EvidenceCategory.CRC_VALIDATION,
            description=f"CRC verification ({crc_type or 'CRC'}): {pass_cnt}/{total_cnt} passed ({ratio*100:.0f}%)",
            value=ratio,
            normalized_score=ratio,
            strength=EvidenceStrength.STRONG if is_pass else (EvidenceStrength.CONTRADICTORY if ratio == 0.0 else EvidenceStrength.WEAK),
            direction=EvidenceDirection.SUPPORT if is_pass else EvidenceDirection.CONTRADICT,
            independence_group=IndependenceGroup.CRC,
            source_reference="astra_validation"
        ))
    elif crc_passed is not None:
        ratio = 1.0 if crc_passed else 0.0
        items.append(create_evidence_item(
            evidence_id="ev_crc_validation",
            stage="Stage 10: Validation Engine",
            category=EvidenceCategory.CRC_VALIDATION,
            description="CRC verification: PASS" if crc_passed else "CRC verification: FAILED / NO PASS",
            value=ratio,
            normalized_score=ratio,
            strength=EvidenceStrength.STRONG if crc_passed else EvidenceStrength.CONTRADICTORY,
            direction=EvidenceDirection.SUPPORT if crc_passed else EvidenceDirection.CONTRADICT,
            independence_group=IndependenceGroup.CRC,
            source_reference="astra_validation"
        ))

    # Validation overall score
    val_score = s10.get("validation_score", record.get("validation_score"))
    if val_score is not None:
        v_sc = float(val_score)
        # When CRC is untested (total_cnt == 0), low validation score reflects missing tests, not contradiction
        if total_cnt == 0 or total_cnt is None and crc_passed is None:
            v_dir = EvidenceDirection.NEUTRAL
            v_str = EvidenceStrength.NEUTRAL
        else:
            v_dir = EvidenceDirection.SUPPORT if v_sc >= 0.50 else EvidenceDirection.CONTRADICT
            v_str = EvidenceStrength.STRONG if v_sc >= 0.85 else EvidenceStrength.MODERATE

        items.append(create_evidence_item(
            evidence_id="ev_validation_score",
            stage="Stage 10: Validation Engine",
            category=EvidenceCategory.CRC_VALIDATION,
            description=f"Stage 10 composite validation score: {v_sc:.2f}",
            value=v_sc,
            normalized_score=v_sc,
            strength=v_str,
            direction=v_dir,
            independence_group=IndependenceGroup.CRC,
            source_reference="astra_validation"
        ))

    # 8. Pipeline Scoring & Margin (Stage 11)
    candidates = s11.get("candidates", [])
    best_pipe = s11.get("best_pipeline_id") or record.get("pipeline_id", "pipeline_best")
    if candidates:
        top_cand = candidates[0]
        p_sc = float(top_cand.get("score", 0.90))
        sec_sc = float(candidates[1].get("score", 0.40)) if len(candidates) > 1 else 0.0
        score_margin = p_sc - sec_sc
        items.append(create_evidence_item(
            evidence_id="ev_pipeline_scorer",
            stage="Stage 11: Pipeline Scorer",
            category=EvidenceCategory.PIPELINE_SCORE,
            description=f"Stage 11 XGBoost rank #1: {best_pipe} (score {p_sc:.2f}, margin {score_margin:.2f})",
            value=p_sc,
            normalized_score=p_sc,
            strength=EvidenceStrength.STRONG if p_sc >= 0.85 and score_margin >= 0.20 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.PIPELINE_RANKING,
            source_reference="astra_pipeline_scorer"
        ))
    elif record.get("pipeline_score") is not None:
        p_sc = float(record.get("pipeline_score"))
        score_margin = float(record.get("score_margin", 0.35))
        items.append(create_evidence_item(
            evidence_id="ev_pipeline_scorer",
            stage="Stage 11: Pipeline Scorer",
            category=EvidenceCategory.PIPELINE_SCORE,
            description=f"Stage 11 XGBoost rank #1: {best_pipe} (score {p_sc:.2f}, margin {score_margin:.2f})",
            value=p_sc,
            normalized_score=p_sc,
            strength=EvidenceStrength.STRONG if p_sc >= 0.85 and score_margin >= 0.20 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.PIPELINE_RANKING,
            source_reference="astra_pipeline_scorer"
        ))

    # 9. Bitstream & Neural Transformer Evidence (Stages 12 & 13)
    frame_len = s12.get("frame_length_bits", record.get("frame_length_bits"))
    autocorr_score = s12.get("periodicity_score", record.get("frame_periodicity_score", 0.92))
    if frame_len is not None:
        items.append(create_evidence_item(
            evidence_id="ev_frame_periodicity",
            stage="Stage 12: Bitstream Intelligence",
            category=EvidenceCategory.FRAME_STRUCTURE,
            description=f"Bitstream periodicity candidate: {frame_len} bits (prominence {autocorr_score:.2f})",
            value=frame_len,
            normalized_score=float(autocorr_score),
            strength=EvidenceStrength.STRONG if autocorr_score >= 0.85 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.FRAME_PERIODICITY,
            source_reference="astra_bitstream_intelligence"
        ))

    regions = s13.get("regions", [])
    if regions:
        pay_region = next((r for r in regions if r.get("region") == "PAYLOAD"), None)
        b_conf = float(pay_region.get("probability", 0.90)) if pay_region else 0.85
        items.append(create_evidence_item(
            evidence_id="ev_stage13_transformer",
            stage="Stage 13: Bitstream Transformer",
            category=EvidenceCategory.BITSTREAM_STRUCTURE,
            description=f"1D CNN + Transformer predicted structural boundaries (payload prob {b_conf:.2f})",
            value=b_conf,
            normalized_score=b_conf,
            strength=EvidenceStrength.STRONG if b_conf >= 0.85 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.HEADER_STRUCTURE,
            source_reference="astra_bitstream_transformer"
        ))
    elif record.get("stage13_boundary_confidence") is not None:
        b_conf = float(record.get("stage13_boundary_confidence"))
        items.append(create_evidence_item(
            evidence_id="ev_stage13_transformer",
            stage="Stage 13: Bitstream Transformer",
            category=EvidenceCategory.BITSTREAM_STRUCTURE,
            description=f"1D CNN + Transformer predicted structural boundaries (confidence {b_conf:.2f})",
            value=b_conf,
            normalized_score=b_conf,
            strength=EvidenceStrength.STRONG if b_conf >= 0.85 else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.HEADER_STRUCTURE,
            source_reference="astra_bitstream_transformer"
        ))

    # 10. Payload Recovery & UTF-8 Text Evidence (Stage 14)
    payload_hex = s14.get("extracted_payload_hex") or record.get("payload_hex")
    payload_utf8 = s14.get("decoded_text") or record.get("payload_utf8")
    payload_bytes_valid = record.get("payload_bytes_valid", True)
    if payload_hex is not None:
        items.append(create_evidence_item(
            evidence_id="ev_payload_bytes",
            stage="Stage 14: Payload Explorer",
            category=EvidenceCategory.PAYLOAD_PARSE,
            description=f"Recovered payload bytes: {len(payload_hex)//2} bytes (hex: {payload_hex[:16]}...)",
            value=payload_hex,
            normalized_score=0.98 if payload_bytes_valid else 0.60,
            strength=EvidenceStrength.STRONG if payload_bytes_valid else EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.PAYLOAD_STRUCTURE,
            source_reference="astra_payload_explorer"
        ))

    if payload_utf8 is not None:
        items.append(create_evidence_item(
            evidence_id="ev_payload_utf8",
            stage="Stage 14: Payload Explorer",
            category=EvidenceCategory.PAYLOAD_PARSE,
            description=f"Payload valid UTF-8 representation: '{payload_utf8}'",
            value=payload_utf8,
            normalized_score=0.95,
            strength=EvidenceStrength.MODERATE,
            direction=EvidenceDirection.SUPPORT,
            independence_group=IndependenceGroup.PAYLOAD_STRUCTURE,
            source_reference="astra_payload_explorer"
        ))

    # 11. User Override Evidence (if present)
    override_fields = record.get("user_overrides", {})
    for f_name, f_val in override_fields.items():
        items.append(create_evidence_item(
            evidence_id=f"ev_override_{f_name}",
            stage="User Interaction (EXPERT Mode)",
            category=EvidenceCategory.USER_OVERRIDE,
            description=f"User override applied for {f_name} = {f_val}",
            value=f_val,
            normalized_score=1.0,
            strength=EvidenceStrength.NEUTRAL,
            direction=EvidenceDirection.NEUTRAL,
            independence_group=IndependenceGroup.USER_OVERRIDE,
            source_reference="expert_user_override"
        ))

    return items
