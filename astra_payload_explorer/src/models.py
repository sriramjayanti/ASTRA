"""
models.py
Data models, enums, dataclasses, and frame structures for ASTRA Stage 14 — Header / Payload Explorer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np


class EvidenceLevel(str, Enum):
    KNOWN = "KNOWN"         # Defined explicitly in a known protocol profile
    INFERRED = "INFERRED"   # Statistically derived from cross-frame analysis
    POSSIBLE = "POSSIBLE"   # Weak heuristic candidate interpretation
    UNKNOWN = "UNKNOWN"     # Uninterpreted raw data


class ExplorerStatus(str, Enum):
    PROFILE_PARSED = "PROFILE_PARSED"
    STRUCTURE_PARSED = "STRUCTURE_PARSED"
    PARTIAL_PARSE = "PARTIAL_PARSE"
    BLIND_EXPLORED = "BLIND_EXPLORED"
    UNKNOWN_PROTOCOL = "UNKNOWN_PROTOCOL"
    INSUFFICIENT_STRUCTURE = "INSUFFICIENT_STRUCTURE"


@dataclass
class FieldDefinition:
    """Schema definition for a protocol field."""
    name: str
    offset_bits: int
    width_bits: int
    field_type: str = "uint"  # uint, int, boolean, bitfield, enum, fixed_point, raw, ascii
    endianness: str = "big"   # big, little
    scale: float = 1.0
    offset: float = 0.0
    unit: Optional[str] = None
    description: Optional[str] = None
    enum_values: Dict[int, str] = field(default_factory=dict)
    subfields: Dict[str, "FieldDefinition"] = field(default_factory=dict)
    expected_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "offset_bits": self.offset_bits,
            "width_bits": self.width_bits,
            "field_type": self.field_type,
            "endianness": self.endianness,
            "scale": self.scale,
            "offset": self.offset,
            "unit": self.unit,
            "description": self.description,
            "enum_values": self.enum_values,
            "expected_value": self.expected_value
        }


@dataclass
class ProtocolProfile:
    """Full specification for a framed protocol profile."""
    profile_id: str
    version: str = "1.0.0"
    schema_version: str = "1.0"
    source: str = "registry"
    description: Optional[str] = None
    frame_length_bits: int = 512
    sync_length_bits: int = 32
    sync_pattern: Optional[str] = None
    sync_offset_bits: int = 0
    header_length_bits: int = 64
    header_offset_bits: int = 32
    crc_length_bits: int = 16
    crc_position: str = "trailer"  # trailer, header, none
    endianness: str = "big"
    bit_order: str = "msb_first"
    header_fields: Dict[str, FieldDefinition] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "schema_version": self.schema_version,
            "source": self.source,
            "frame_length_bits": self.frame_length_bits,
            "sync_length_bits": self.sync_length_bits,
            "sync_pattern": self.sync_pattern,
            "header_length_bits": self.header_length_bits,
            "crc_length_bits": self.crc_length_bits,
            "endianness": self.endianness,
            "bit_order": self.bit_order,
            "header_fields": {k: v.to_dict() for k, v in self.header_fields.items()}
        }


@dataclass
class ParsedField:
    """Represents an individual parsed or discovered header field."""
    field_name: str
    offset_bits: int
    width_bits: int
    raw_bits: str = ""
    raw_hex: str = ""
    decoded_value: Any = None
    raw_unsigned: Optional[int] = None
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    source: str = "blind_exploration"
    unit: Optional[str] = None
    description: Optional[str] = None
    valid_constraint: bool = True
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": str(self.field_name),
            "offset_bits": int(self.offset_bits),
            "width_bits": int(self.width_bits),
            "raw_hex": str(self.raw_hex),
            "decoded_value": self.decoded_value,
            "raw_unsigned": self.raw_unsigned,
            "evidence_level": self.evidence_level.value if isinstance(self.evidence_level, EvidenceLevel) else str(self.evidence_level),
            "source": str(self.source),
            "unit": self.unit,
            "description": self.description,
            "valid_constraint": self.valid_constraint,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class RegionInfo:
    """Generic structural region within a bitstream / frame."""
    start_bit: int
    end_bit: int
    length_bits: int
    raw_bits: Optional[np.ndarray] = None
    raw_hex: str = ""
    label: str = "UNKNOWN"
    confidence: float = 1.0
    support_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_bit": self.start_bit,
            "end_bit": self.end_bit,
            "length_bits": self.length_bits,
            "raw_hex": self.raw_hex,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "support_sources": self.support_sources
        }


@dataclass
class SyncRegionInfo:
    """Parsed sync/preamble region within a frame."""
    raw_bits: str = ""
    raw_hex: str = ""
    start_bit: int = 0
    end_bit: int = 0
    length_bits: int = 0
    matched_pattern: Optional[str] = None
    hamming_distance: int = 0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_hex": self.raw_hex,
            "start_bit": self.start_bit,
            "end_bit": self.end_bit,
            "length_bits": self.length_bits,
            "matched_pattern": self.matched_pattern,
            "hamming_distance": self.hamming_distance,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class CRCRegionInfo:
    """CRC / Checksum verification telemetry for a frame."""
    raw_bits: str = ""
    raw_hex: str = ""
    start_bit: int = 0
    end_bit: int = 0
    length_bits: int = 0
    observed_value: Optional[int] = None
    expected_value: Optional[int] = None
    validation_status: str = "UNKNOWN"  # PASS, FAIL, UNCHECKED, UNKNOWN
    polynomial: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_hex": self.raw_hex,
            "start_bit": self.start_bit,
            "end_bit": self.end_bit,
            "length_bits": self.length_bits,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "validation_status": self.validation_status,
            "polynomial": self.polynomial,
        }


@dataclass
class PayloadViews:
    """Multi-view representation of extracted payload content without modifying source bits."""
    length_bits: int = 0
    length_bytes: int = 0
    hex_str: str = ""
    raw_bits: str = ""
    byte_array: List[int] = field(default_factory=list)
    byte_aligned: bool = True
    remainder_bits: int = 0
    representations: Dict[str, Any] = field(default_factory=dict)
    entropy_shannon: float = 0.0
    file_signatures: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_raw_bits: bool = False) -> Dict[str, Any]:
        d = {
            "hex_str": self.hex_str,
            "length_bits": self.bit_length,
            "length_bytes": self.byte_count,
            "byte_aligned": self.is_byte_aligned,
            "byte_array": self.byte_array[:64],
            "representations": self.representations,
            "entropy": round(float(self.entropy_shannon), 4),
            "file_signatures": self.file_signatures,
        }
        if include_raw_bits:
            d["raw_bits"] = self.raw_bits
        return d

    @property
    def byte_count(self) -> int:
        return self.length_bytes

    @property
    def bit_length(self) -> int:
        return self.length_bits

    @property
    def is_byte_aligned(self) -> bool:
        return self.byte_aligned


@dataclass
class FrameCandidate:
    """Candidate hypothesis for frame length and alignment offset."""
    length_bits: int
    alignment_offset_bits: int = 0
    support_score: float = 1.0
    stage12_score: float = 0.0
    stage13_consistency: float = 0.0
    validation_support: float = 0.0
    frame_count: int = 1
    support_sources: List[str] = field(default_factory=list)
    protocol_profile_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length_bits": self.length_bits,
            "alignment_offset_bits": self.alignment_offset_bits,
            "support_score": round(float(self.support_score), 4),
            "support_sources": self.support_sources,
            "protocol_profile_id": self.protocol_profile_id
        }


# Alias for compatibility
CandidateInterpretation = FrameCandidate


@dataclass
class FrameRecord:
    """Complete segmented and parsed record for an individual frame."""
    frame_index: int
    start_bit: int
    end_bit: int
    frame_length_bits: int

    # Structural Regions
    sync_region: Optional[Union[SyncRegionInfo, RegionInfo]] = None
    header_region: Optional[RegionInfo] = None
    payload_region: Optional[RegionInfo] = None
    crc_region: Optional[Union[CRCRegionInfo, RegionInfo]] = None
    padding_region: Optional[RegionInfo] = None
    unknown_regions: List[RegionInfo] = field(default_factory=list)

    # Parsed Content
    header_fields: Dict[str, ParsedField] = field(default_factory=dict)
    payload: Optional[PayloadViews] = None
    crc_metadata: Optional[CRCRegionInfo] = None

    # Status & Hashes
    is_partial: bool = False
    is_corrupted: bool = False
    frame_quality_score: float = 1.0
    boundary_confidence: float = 1.0
    frame_hash: str = ""
    payload_hash: str = ""
    header_hash: str = ""

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "start_bit": self.start_bit,
            "end_bit": self.end_bit,
            "frame_length_bits": self.frame_length_bits,
            "is_partial": self.is_partial,
            "is_corrupted": self.is_corrupted,
            "frame_quality_score": round(float(self.frame_quality_score), 4),
            "boundary_confidence": round(float(self.boundary_confidence), 4),
            "sync": self.sync_region.to_dict() if self.sync_region and hasattr(self.sync_region, "to_dict") else None,
            "header_fields": {k: v.to_dict() for k, v in self.header_fields.items()},
            "payload": self.payload.to_dict(include_raw_bits=include_raw) if self.payload else None,
            "crc": self.crc_region.to_dict() if self.crc_region and hasattr(self.crc_region, "to_dict") else None,
            "frame_hash": self.frame_hash,
            "payload_hash": self.payload_hash,
            "header_hash": self.header_hash
        }


@dataclass
class BlindDiscoveredField:
    """Discovered candidate field from blind exploratory analysis."""
    field_name: str
    offset_bits: int
    width_bits: int
    endianness: str = "big"
    values_across_frames: List[Any] = field(default_factory=list)
    pattern: str = "unknown"  # constant, incrementing, length_correlating, variable
    possible_role: str = "unknown"  # constant_field_candidate, counter_candidate, length_candidate
    confidence: float = 0.5
    evidence_level: EvidenceLevel = EvidenceLevel.INFERRED
    correlation: float = 0.0
    exact_match_fraction: float = 0.0
    wrap_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "offset_bits": self.offset_bits,
            "width_bits": self.width_bits,
            "endianness": self.endianness,
            "values_across_frames": self.values_across_frames[:10],
            "pattern": self.pattern,
            "possible_role": self.possible_role,
            "confidence": round(float(self.confidence), 4),
            "evidence_level": self.evidence_level.value,
            "correlation": round(float(self.correlation), 4),
        }


@dataclass
class PayloadExplorerContext:
    """Context and inputs passed to the Header / Payload Explorer."""
    pipeline_path_id: str = "pipeline_default"
    stage12_result: Optional[Any] = None
    stage13_result: Optional[Any] = None
    validation_result: Optional[Any] = None
    protocol_profiles: Dict[str, ProtocolProfile] = field(default_factory=dict)
    known_profile_id: Optional[str] = None
    bit_confidence: Optional[np.ndarray] = None
    mode: str = "auto"  # auto, profile_aware, blind_exploration


@dataclass
class PayloadExplorerResult:
    """
    Comprehensive output of ASTRA Stage 14 Header / Payload Explorer.
    Contains parsed frame records, decoded payload views, discovered fields, and provenance.
    """
    pipeline_path_id: str
    frame_count: int
    frames: List[FrameRecord] = field(default_factory=list)

    selected_frame_length: int = 512
    selected_alignment: int = 0
    protocol_profile_used: Optional[str] = None
    profile_match_score: float = 0.0

    global_payload_summary: Dict[str, Any] = field(default_factory=dict)
    blind_discovered_fields: List[BlindDiscoveredField] = field(default_factory=list)
    candidate_interpretations: List[FrameCandidate] = field(default_factory=list)

    explorer_status: ExplorerStatus = ExplorerStatus.BLIND_EXPLORED
    processing_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_frame_details: bool = True) -> Dict[str, Any]:
        data = {
            "pipeline_path_id": str(self.pipeline_path_id),
            "explorer_status": self.explorer_status.value,
            "frame_count": int(self.frame_count),
            "selected_frame_length": int(self.selected_frame_length),
            "selected_alignment": int(self.selected_alignment),
            "protocol_profile_used": self.protocol_profile_used,
            "profile_match_score": round(float(self.profile_match_score), 4),
            "global_payload_summary": self.global_payload_summary,
            "blind_discovered_fields": [f.to_dict() for f in self.blind_discovered_fields],
            "candidate_interpretations": [c.to_dict() for c in self.candidate_interpretations[:5]],
            "processing_history": self.processing_history,
        }
        if include_frame_details:
            data["frames"] = [f.to_dict() for f in self.frames[:20]]
            if len(self.frames) > 20:
                data["frames_truncated_note"] = f"Displaying first 20 of {len(self.frames)} frames."
        return data
