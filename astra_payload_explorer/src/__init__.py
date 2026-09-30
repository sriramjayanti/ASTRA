"""
ASTRA Stage 14 — Header / Payload Explorer Package.
"""

from .models import (
    EvidenceLevel,
    ExplorerStatus,
    FieldDefinition,
    ProtocolProfile,
    ParsedField,
    RegionInfo,
    SyncRegionInfo,
    CRCRegionInfo,
    PayloadViews,
    FrameCandidate,
    CandidateInterpretation,
    FrameRecord,
    BlindDiscoveredField,
    PayloadExplorerContext,
    PayloadExplorerResult
)
from .frame_resolver import resolve_frame_hypotheses, select_frame_segmentation
from .segmentation import slice_frames, compute_bit_hash
from .header_parser import parse_header, parse_field_value
from .blind_fields import (
    discover_constant_fields,
    discover_counter_candidates,
    discover_length_candidates,
    analyze_field_variability
)
from .payload_decoders import (
    decode_payload_views,
    decode_text_representations,
    detect_file_signatures,
    decode_base64
)
from .byte_alignment import (
    bits_to_bytes,
    bytes_to_bits,
    bits_to_hex_str,
    hex_to_bits
)
from .bit_order import (
    bits_to_uint,
    bits_to_int,
    uint_to_bits,
    int_to_bits,
    reverse_bits_in_bytes
)
from .endian import (
    decode_integer_with_endian,
    decode_fixed_point,
    encode_integer_with_endian
)
from .cross_frame import (
    detect_duplicate_frames,
    analyze_payload_variations,
    check_sequence_continuity,
    analyze_cross_frame_statistics
)
from .confidence import (
    compute_frame_quality_score,
    compute_boundary_confidence
)
from .exporters import (
    export_to_json,
    export_to_csv,
    generate_hex_dump,
    export_binary_payload
)
from .validators import score_profile_match
from .inference import HeaderPayloadExplorer

__all__ = [
    "EvidenceLevel",
    "ExplorerStatus",
    "FieldDefinition",
    "ProtocolProfile",
    "ParsedField",
    "RegionInfo",
    "SyncRegionInfo",
    "CRCRegionInfo",
    "PayloadViews",
    "FrameCandidate",
    "CandidateInterpretation",
    "FrameRecord",
    "BlindDiscoveredField",
    "PayloadExplorerContext",
    "PayloadExplorerResult",
    "resolve_frame_hypotheses",
    "select_frame_segmentation",
    "slice_frames",
    "compute_bit_hash",
    "parse_header",
    "parse_field_value",
    "discover_constant_fields",
    "discover_counter_candidates",
    "discover_length_candidates",
    "analyze_field_variability",
    "decode_payload_views",
    "decode_text_representations",
    "detect_file_signatures",
    "decode_base64",
    "bits_to_bytes",
    "bytes_to_bits",
    "bits_to_hex_str",
    "hex_to_bits",
    "bits_to_uint",
    "bits_to_int",
    "uint_to_bits",
    "int_to_bits",
    "reverse_bits_in_bytes",
    "decode_integer_with_endian",
    "decode_fixed_point",
    "encode_integer_with_endian",
    "detect_duplicate_frames",
    "analyze_payload_variations",
    "check_sequence_continuity",
    "analyze_cross_frame_statistics",
    "compute_frame_quality_score",
    "compute_boundary_confidence",
    "export_to_json",
    "export_to_csv",
    "generate_hex_dump",
    "export_binary_payload",
    "score_profile_match",
    "HeaderPayloadExplorer",
]
