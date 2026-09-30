"""
contradictions.py
Contradiction analysis and inconsistency detection across multi-stage pipeline evidence.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .models import (
    Contradiction,
    ContradictionSeverity,
    EvidenceItem,
    EvidenceCategory,
    EvidenceDirection
)


class ContradictionAnalyzer:
    """
    Detects physical, mathematical, and model contradictions across pipeline stages.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        penalties = self.config.get("contradiction_penalties", {})
        self.penalties = {
            ContradictionSeverity.LOW: penalties.get("low", 0.02),
            ContradictionSeverity.MEDIUM: penalties.get("medium", 0.06),
            ContradictionSeverity.HIGH: penalties.get("high", 0.15),
            ContradictionSeverity.CRITICAL: penalties.get("critical", 0.30)
        }

    def analyze(self, record: Dict[str, Any], evidence_items: List[EvidenceItem]) -> List[Contradiction]:
        """Examine record and evidence items for conflicting signals."""
        contradictions: List[Contradiction] = []

        # 1. Model vs Constellation Inconsistency
        s3 = record.get("stage_3_fusion", {})
        s4 = record.get("stage_4_constellation", {})
        s6 = record.get("stage_6_sync", {})
        s7 = record.get("stage_7_demod", {})
        s9 = record.get("stage_9_fec", {})
        s10 = record.get("stage_10_validation", {})
        s12 = record.get("stage_12_bitstream", {})
        s14 = record.get("stage_14_payload", {})

        mod = str(s3.get("predicted_class") or record.get("fusion_modulation") or record.get("modulation", "")).upper()
        clusters = s4.get("num_clusters") or record.get("constellation_clusters")
        if "QPSK" in mod and clusters in [16, 64]:
            c = Contradiction(
                contradiction_id="contra_mod_vs_constellation",
                fields_involved=["modulation", "constellation"],
                evidence_a=f"Modulation model proposed {mod}",
                evidence_b=f"Post-sync constellation exhibits {clusters} clusters",
                severity=ContradictionSeverity.HIGH,
                description="Modulation model selected 4-ary PSK but constellation geometry indicates a high-order QAM scheme.",
                resolution_status="UNRESOLVED",
                penalty=self.penalties[ContradictionSeverity.HIGH]
            )
            contradictions.append(c)

        # 2. Timing Recovery Failure with High Early Baud Score
        baud_score = record.get("baud_score", 0.0)
        timing_lock = s6.get("timing_lock_metric", record.get("timing_lock_metric"))
        if baud_score >= 0.85 and timing_lock is not None and timing_lock < 0.40:
            c = Contradiction(
                contradiction_id="contra_baud_vs_timing_lock",
                fields_involved=["symbol_rate", "synchronization"],
                evidence_a=f"Symbol rate candidate score was high ({baud_score:.2f})",
                evidence_b=f"Timing recovery failed to achieve lock (metric: {timing_lock:.2f})",
                severity=ContradictionSeverity.MEDIUM,
                description="Candidate baud rate scored high in DSP analysis but Gardner/M&M timing recovery failed to achieve stable symbol lock.",
                resolution_status="UNRESOLVED",
                penalty=self.penalties[ContradictionSeverity.MEDIUM]
            )
            contradictions.append(c)

        # 3. FEC Decoded / Valid but Repeated CRC Failures
        fec_match = s9.get("reencoding_agreement_pct") or record.get("fec_reencoding_match_ratio")
        if fec_match is not None and fec_match > 1.0:
            fec_match = fec_match / 100.0
        crc_passed = s10.get("crc_passed", record.get("crc_passed"))
        pass_cnt = s10.get("crc_pass_count")
        tot_cnt = s10.get("crc_total_checked")
        crc_ratio = (pass_cnt / tot_cnt) if tot_cnt else record.get("crc_pass_ratio")

        if fec_match is not None and fec_match >= 0.95:
            if (crc_passed is False) or (crc_ratio is not None and crc_ratio == 0.0 and tot_cnt and tot_cnt > 0):
                c = Contradiction(
                    contradiction_id="contra_fec_vs_crc",
                    fields_involved=["fec", "crc_validation"],
                    evidence_a=f"FEC re-encoding match is near perfect ({fec_match*100:.1f}%)",
                    evidence_b="Downstream CRC checksum checks all failed (0% pass)",
                    severity=ContradictionSeverity.HIGH,
                    description="FEC decoder appeared to converge, but recovered bitstream completely failed CRC checksum verification. Possible wrong interleaver, inverted sync, or corrupt parity bits.",
                    resolution_status="UNRESOLVED",
                    penalty=self.penalties[ContradictionSeverity.HIGH]
                )
                contradictions.append(c)

        # 4. Header Declared Length vs Physical Frame Boundaries
        header_len_field = s14.get("parsed_fields", {}).get("length") or record.get("header_declared_length_bytes")
        extracted_len_bytes = s14.get("payload_length_bytes") or record.get("extracted_payload_bytes")
        if header_len_field is not None and extracted_len_bytes is not None:
            if abs(header_len_field - extracted_len_bytes) > 2:
                c = Contradiction(
                    contradiction_id="contra_header_vs_boundary",
                    fields_involved=["header", "frame_structure"],
                    evidence_a=f"Header length field declared {header_len_field} bytes",
                    evidence_b=f"Physical frame boundary isolated {extracted_len_bytes} bytes",
                    severity=ContradictionSeverity.MEDIUM,
                    description=f"Mismatch of {abs(header_len_field - extracted_len_bytes)} bytes between header-declared payload size and observed physical frame boundary.",
                    resolution_status="UNRESOLVED",
                    penalty=self.penalties[ContradictionSeverity.MEDIUM]
                )
                contradictions.append(c)

        # 5. User Override Conflict
        user_overrides = record.get("user_overrides", {})
        if user_overrides:
            for field_name, user_val in user_overrides.items():
                auto_val = s3.get("predicted_class") or record.get(field_name) or record.get(f"{field_name}_modulation")
                if auto_val is not None and str(user_val).upper() != str(auto_val).upper():
                    c = Contradiction(
                        contradiction_id=f"contra_user_override_{field_name}",
                        fields_involved=[field_name, "user_override"],
                        evidence_a=f"User forced manual override: {user_val}",
                        evidence_b=f"Pipeline models and measurements favored: {auto_val}",
                        severity=ContradictionSeverity.HIGH,
                        description=f"User override hypothesis ({user_val}) directly conflicts with pipeline model consensus and downstream evidence ({auto_val}).",
                        resolution_status="OVERRIDDEN_BY_USER",
                        penalty=self.penalties[ContradictionSeverity.HIGH]
                    )
                    contradictions.append(c)

        # 6. Demodulation EVM vs Downstream Success
        evm = s7.get("evm") or record.get("demod_evm_pct")
        if evm is not None and evm > 1.0:
            evm_pct = evm
        elif evm is not None:
            evm_pct = evm * 100.0
        else:
            evm_pct = None
        if evm_pct is not None and evm_pct > 35.0:
            if crc_passed is True or (crc_ratio is not None and crc_ratio >= 0.80):
                # Resolved contradiction: high EVM on channel, but FEC cleaned it up!
                c = Contradiction(
                    contradiction_id="contra_high_evm_but_crc_pass",
                    fields_involved=["demodulation", "crc_validation"],
                    evidence_a=f"High demodulation EVM ({evm_pct:.1f}%) suggests noisy symbols",
                    evidence_b="Downstream CRC checksum checks passed successfully",
                    severity=ContradictionSeverity.LOW,
                    description="Channel had severe noise or distortion (high EVM), but FEC successfully corrected symbol errors to yield valid CRC frames.",
                    resolution_status="RESOLVED_BY_DOWNSTREAM",
                    penalty=self.penalties[ContradictionSeverity.LOW] * 0.5
                )
                contradictions.append(c)

        return contradictions

    def compute_total_penalty(self, contradictions: List[Contradiction]) -> float:
        """Compute cumulative confidence penalty from unresolved contradictions."""
        penalty = 0.0
        for c in contradictions:
            if c.resolution_status != "RESOLVED_BY_DOWNSTREAM":
                penalty += c.penalty
        return float(np.clip(penalty, 0.0, 0.40))
