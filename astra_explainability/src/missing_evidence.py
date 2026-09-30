"""
missing_evidence.py
Identification and impact assessment of absent telemetry or unapplied validation tests.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .models import MissingEvidence, EvidenceCategory


class MissingEvidenceAnalyzer:
    """
    Analyzes what evidence was absent during the signal recovery workflow.
    Rule: Missing evidence reduces confidence appropriately, but is NOT failure.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        penalties = self.config.get("missing_evidence_penalties", {})
        self.penalties = {
            "missing_crc": penalties.get("missing_crc", 0.08),
            "missing_profile": penalties.get("missing_profile", 0.03),
            "short_capture": penalties.get("short_capture", 0.05),
            "single_frame": penalties.get("single_frame", 0.05)
        }

    def analyze(self, record: Dict[str, Any]) -> List[MissingEvidence]:
        """Audit record for missing evidence categories."""
        missing: List[MissingEvidence] = []

        s10 = record.get("stage_10_validation", {})
        s14 = record.get("stage_14_payload", {})

        # 1. CRC Checksum Missing
        crc_passed = s10.get("crc_passed", record.get("crc_passed"))
        crc_ratio = s10.get("crc_pass_ratio", record.get("crc_pass_ratio"))
        tot_checked = s10.get("crc_total_checked")
        if (crc_passed is None and crc_ratio is None) or (tot_checked == 0):
            missing.append(MissingEvidence(
                missing_id="miss_crc_untested",
                category=EvidenceCategory.CRC_VALIDATION,
                field_name="crc_validation",
                impact_description="No known CRC polynomial or check region was defined. Bitstream validity relies on FEC convergence and frame periodicity.",
                evidence_type="CRC Validation",
                severity="MEDIUM",
                penalty=self.penalties["missing_crc"]
            ))

        # 2. Known Protocol Profile Missing
        profile_used = s14.get("protocol_profile") or record.get("protocol_profile_used")
        if not profile_used:
            missing.append(MissingEvidence(
                missing_id="miss_protocol_profile",
                category=EvidenceCategory.FRAME_STRUCTURE,
                field_name="protocol",
                impact_description="No matching protocol profile in registry. Header and payload were analyzed via blind exploratory parsing.",
                evidence_type="Protocol Profile",
                severity="LOW",
                penalty=self.penalties["missing_profile"]
            ))

        # 3. Short Capture / Constellation Limitation
        sample_count = record.get("sample_count", 2048)
        if sample_count < 1024:
            missing.append(MissingEvidence(
                missing_id="miss_short_capture",
                category=EvidenceCategory.CONSTELLATION,
                field_name="constellation",
                impact_description=f"Capture length ({sample_count} samples) too short for high-density constellation clustering or multi-ring DBSCAN analysis.",
                severity="LOW",
                penalty=self.penalties["short_capture"]
            ))

        # 4. Single Frame / Insufficient Frames for Cross-Frame Counters
        frame_count = record.get("frame_count", 1)
        if frame_count <= 1:
            missing.append(MissingEvidence(
                missing_id="miss_multiple_frames",
                category=EvidenceCategory.FRAME_STRUCTURE,
                field_name="frame_structure",
                impact_description="Only 1 frame recovered in burst. Cross-frame counter progression and payload variation could not be evaluated.",
                severity="LOW",
                penalty=self.penalties["single_frame"]
            ))

        return missing

    def compute_total_penalty(self, missing_list: List[MissingEvidence]) -> float:
        """Compute cumulative penalty from missing evidence."""
        penalty = sum(m.penalty for m in missing_list)
        return float(np.clip(penalty, 0.0, 0.20))
