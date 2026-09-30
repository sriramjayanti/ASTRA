"""
inference.py
Main Header / Payload Explorer engine orchestrator for ASTRA Stage 14.
"""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import os
import yaml
import numpy as np

from .models import (
    EvidenceLevel,
    ExplorerStatus,
    ParsedField,
    SyncRegionInfo,
    CRCRegionInfo,
    PayloadViews,
    FrameRecord,
    FrameCandidate,
    BlindDiscoveredField,
    ProtocolProfile,
    FieldDefinition,
    PayloadExplorerContext,
    PayloadExplorerResult
)
from .frame_resolver import resolve_frame_hypotheses, select_frame_segmentation
from .segmentation import slice_frames, compute_bit_hash
from .validators import score_profile_match
from .header_parser import parse_header, parse_field_value
from .blind_fields import (
    discover_constant_fields,
    discover_counter_candidates,
    discover_length_candidates,
    analyze_field_variability
)
from .payload_decoders import decode_payload_views
from .cross_frame import analyze_cross_frame_statistics
from .confidence import compute_frame_quality_score
from .byte_alignment import bits_to_hex_str


class HeaderPayloadExplorer:
    """
    ASTRA Stage 14: Header / Payload Explorer Engine.
    Segments recovered bitstreams into frames, parses known and blind header fields,
    extracts payload in multiple non-destructive representations, and evaluates cross-frame consistency.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.profiles: Dict[str, ProtocolProfile] = self._load_profiles()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_config = {
            "version": "1.0.0",
            "mode": "auto",
            "frames": {"max_candidates": 5, "default_frame_length": 512, "allow_partial_frames": True},
            "profiles": {"enabled": True, "minimum_match_score": 0.75},
            "blind_fields": {"enabled": True, "candidate_widths": [4, 8, 12, 16, 24, 32]},
            "preview": {"max_preview_bytes": 64}
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and "payload_explorer" in loaded:
                    return loaded["payload_explorer"]
                elif loaded:
                    return loaded

        pkg_conf = Path(__file__).parent.parent / "configs" / "payload_explorer_config.yaml"
        if pkg_conf.exists():
            with open(pkg_conf, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and "payload_explorer" in loaded:
                    return loaded["payload_explorer"]

        return default_config

    def _load_profiles(self) -> Dict[str, ProtocolProfile]:
        profiles = {}
        prof_file = Path(__file__).parent.parent / "configs" / "protocol_profiles.yaml"
        if prof_file.exists():
            with open(prof_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "profiles" in data:
                    for p_data in data["profiles"]:
                        pid = p_data.get("profile_id", "prof")
                        p_frame = p_data.get("frame", {})
                        p_sync = p_data.get("sync", {})
                        p_hdr = p_data.get("header", {})
                        p_crc = p_data.get("crc", {})

                        fields = {}
                        for f_dict in p_hdr.get("fields", []):
                            fname = f_dict.get("name", "field")
                            fields[fname] = FieldDefinition(
                                name=fname,
                                offset_bits=f_dict.get("offset_bits", 0),
                                width_bits=f_dict.get("width_bits", 8),
                                field_type=f_dict.get("type", "uint"),
                                endianness=f_dict.get("endian", "big"),
                                scale=f_dict.get("scale", 1.0),
                                offset=f_dict.get("offset", 0.0),
                                unit=f_dict.get("unit"),
                                description=f_dict.get("description"),
                                expected_value=f_dict.get("expected_value")
                            )

                        prof = ProtocolProfile(
                            profile_id=pid,
                            version=p_data.get("version", "1.0.0"),
                            frame_length_bits=p_frame.get("length_bits", 512),
                            sync_length_bits=p_sync.get("length_bits", 32),
                            sync_pattern=p_sync.get("pattern_hex") or p_sync.get("pattern_bits"),
                            sync_offset_bits=p_sync.get("offset_bits", 0),
                            header_length_bits=p_hdr.get("length_bits", 32),
                            crc_length_bits=p_crc.get("width_bits", 16),
                            header_fields=fields
                        )
                        profiles[pid] = prof
        return profiles

    def explore(
        self,
        decoded_bits: np.ndarray,
        context: Optional[Union[PayloadExplorerContext, Dict[str, Any]]] = None
    ) -> PayloadExplorerResult:
        """
        Main analysis entrypoint for header parsing and payload extraction.
        """
        # Normalize context
        if context is None:
            ctx = PayloadExplorerContext()
        elif isinstance(context, dict):
            ctx = PayloadExplorerContext(
                pipeline_path_id=context.get("pipeline_path_id", "pipeline_default"),
                stage12_result=context.get("stage12_result"),
                stage13_result=context.get("stage13_result"),
                validation_result=context.get("validation_result"),
                protocol_profiles=context.get("protocol_profiles", self.profiles),
                known_profile_id=context.get("known_profile_id"),
                mode=context.get("mode", self.config.get("mode", "auto"))
            )
        else:
            ctx = context

        pipeline_path_id = ctx.pipeline_path_id
        active_profiles = {**self.profiles, **ctx.protocol_profiles}
        forced_mode = ctx.mode

        n_bits = len(decoded_bits)
        if n_bits == 0:
            return PayloadExplorerResult(
                pipeline_path_id=pipeline_path_id,
                frame_count=0,
                explorer_status=ExplorerStatus.INSUFFICIENT_STRUCTURE
            )

        # 1. Profile Matching / Selection
        selected_profile = None
        best_profile_score = 0.0
        best_profile_id = None

        if ctx.known_profile_id and ctx.known_profile_id in active_profiles:
            selected_profile = active_profiles[ctx.known_profile_id]
            best_profile_score = 1.0
            best_profile_id = ctx.known_profile_id
        elif forced_mode in ["auto", "profile_aware"] and self.config.get("profiles", {}).get("enabled", True):
            for pid, prof in active_profiles.items():
                score, _, _ = score_profile_match(decoded_bits, prof)
                if score > best_profile_score:
                    best_profile_score = score
                    best_profile_id = pid
                    if score >= self.config.get("profiles", {}).get("minimum_match_score", 0.75):
                        selected_profile = prof

        # 2. Resolve Frame Length & Alignment
        candidates = resolve_frame_hypotheses(
            decoded_bits=decoded_bits,
            stage12_result=ctx.stage12_result,
            stage13_result=ctx.stage13_result,
            validation_result=ctx.validation_result,
            profile=selected_profile,
            default_frame_length=self.config.get("frames", {}).get("default_frame_length", 512)
        )

        top_cand = select_frame_segmentation(candidates)
        if top_cand is None:
            top_cand = FrameCandidate(length_bits=512, alignment_offset_bits=0)

        # 3. Physical Frame Slicing
        frame_records = slice_frames(
            stream_bits=decoded_bits,
            candidate=top_cand,
            profile=selected_profile,
            allow_partial=self.config.get("frames", {}).get("allow_partial_frames", True)
        )

        # 4. Header Parsing (Known or Blind)
        for f in frame_records:
            if not f.is_partial and f.header_region is not None and f.header_region.raw_bits is not None:
                if selected_profile and forced_mode in ["auto", "profile_aware"]:
                    f.header_fields = parse_header(f.header_region.raw_bits, selected_profile)

        # 5. Blind Exploratory Field Discovery
        blind_fields: List[BlindDiscoveredField] = []
        if (not selected_profile or forced_mode == "blind_exploration") and len(frame_records) >= 2:
            consts = discover_constant_fields(frame_records)
            counters = discover_counter_candidates(frame_records)
            lengths = discover_length_candidates(frame_records)
            blind_fields = consts + counters + lengths

            # Populate candidate header fields in frames
            for f in frame_records:
                if not f.header_fields and not f.is_partial and f.header_region is not None and f.header_region.raw_bits is not None:
                    h_bits = f.header_region.raw_bits
                    for bf in blind_fields:
                        b_off = bf.offset_bits
                        b_w = bf.width_bits
                        if b_off + b_w <= len(h_bits):
                            b_sub = h_bits[b_off:b_off + b_w]
                            val = int(np.dot(b_sub, 2 ** np.arange(b_w - 1, -1, -1)))
                            f.header_fields[bf.field_name] = ParsedField(
                                field_name=bf.field_name,
                                offset_bits=b_off,
                                width_bits=b_w,
                                raw_bits="".join(str(b) for b in b_sub),
                                raw_hex=bits_to_hex_str(b_sub),
                                decoded_value=val,
                                evidence_level=EvidenceLevel.INFERRED,
                                source="cross_frame_inference",
                                description=bf.possible_role,
                                confidence=bf.confidence
                            )

        # 6. Status Determination
        if selected_profile and best_profile_score >= 0.75:
            status = ExplorerStatus.PROFILE_PARSED
        elif len(frame_records) > 0 and blind_fields:
            status = ExplorerStatus.BLIND_EXPLORED
        elif len(frame_records) > 0:
            status = ExplorerStatus.STRUCTURE_PARSED
        else:
            status = ExplorerStatus.UNKNOWN_PROTOCOL

        # 7. Cross-Frame and Global Summary
        cross_stats = analyze_cross_frame_statistics(frame_records)
        global_summary = {
            "total_frames_parsed": len(frame_records),
            "cross_frame_stats": cross_stats,
            "sample_payload_utf8": frame_records[0].payload.representations.get("utf8", {}).get("value") if (frame_records and frame_records[0].payload) else None,
            "sample_payload_hex": frame_records[0].payload.hex_str[:32] if (frame_records and frame_records[0].payload) else None
        }

        history_entry = {
            "stage": "header_payload_explorer",
            "status": status.value,
            "profile_used": best_profile_id if selected_profile else None,
            "profile_score": round(best_profile_score, 4),
            "frame_length": top_cand.length_bits,
            "frames_count": len(frame_records)
        }

        return PayloadExplorerResult(
            pipeline_path_id=pipeline_path_id,
            frame_count=len(frame_records),
            frames=frame_records,
            selected_frame_length=top_cand.length_bits,
            selected_alignment=top_cand.alignment_offset_bits,
            protocol_profile_used=best_profile_id if selected_profile else None,
            profile_match_score=best_profile_score,
            global_payload_summary=global_summary,
            blind_discovered_fields=blind_fields,
            candidate_interpretations=candidates,
            explorer_status=status,
            processing_history=[history_entry]
        )

    # Convenience API helpers
    def explore_frame(self, frame_bits: np.ndarray, profile: Optional[ProtocolProfile] = None) -> FrameRecord:
        """Parse a single isolated frame."""
        cand = FrameCandidate(length_bits=len(frame_bits), alignment_offset_bits=0)
        frames = slice_frames(frame_bits, cand, profile, allow_partial=True)
        if frames:
            f = frames[0]
            if profile and f.header_region and f.header_region.raw_bits is not None:
                f.header_fields = parse_header(f.header_region.raw_bits, profile)
            return f
        return FrameRecord(frame_index=0, start_bit=0, end_bit=len(frame_bits), frame_length_bits=len(frame_bits))

    def parse_profile(self, header_bits: np.ndarray, profile: ProtocolProfile) -> Dict[str, ParsedField]:
        """Convenience method for profile header parsing."""
        return parse_header(header_bits, profile)

    def blind_explore_header(self, frames: List[FrameRecord]) -> List[BlindDiscoveredField]:
        """Convenience method for blind header exploration across frames."""
        return discover_constant_fields(frames) + discover_counter_candidates(frames) + discover_length_candidates(frames)

    def decode_payload(self, payload_bits: np.ndarray) -> PayloadViews:
        """Convenience method for non-destructive payload decoding."""
        return decode_payload_views(payload_bits)
