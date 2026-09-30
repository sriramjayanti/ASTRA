"""
ASTRA Stage 10 — Validation Engine package.
"""

from .models import (
    ValidationStatus,
    EvidenceCheckState,
    CRCProfile,
    CRCResult,
    ParityResult,
    SyndromeResult,
    ReencodingResult,
    FrameRepetitionResult,
    SyncWordResult,
    HeaderConsistencyResult,
    LengthConsistencyResult,
    StructuralResult,
    ValidationResult,
)

from .crc import (
    CRCCalculator,
    compute_crc,
    check_crc_frame,
    search_crc_candidates,
    load_crc_profiles,
    reflect_bits,
    bits_to_bytes,
)

from .parity import (
    check_even_parity,
    check_odd_parity,
    check_block_parity,
)

from .syndrome import normalize_syndrome_evidence
from .repetition import (
    estimate_repetition_periods,
    score_frame_repetition,
    segment_by_period,
)

from .sync_word import (
    search_sync_word,
    score_sync_periodicity,
    hex_to_bits,
)

from .headers import (
    validate_constant_field,
    validate_monotonic_counter,
    validate_header_schema,
)

from .length_checks import validate_length_field
from .structure import extract_structural_evidence
from .decoder_support import reencode_and_compare
from .scoring import (
    evaluate_evidence_groups,
    compute_overall_validation_score,
    determine_validation_status,
    extract_stage11_features,
    VALIDATION_FEATURE_SCHEMA_VERSION,
)
from .inference import ValidationEngine
from .utils import (
    generate_synthetic_frame,
    generate_synthetic_stream,
    compute_bitstream_hash,
)

__all__ = [
    "ValidationStatus",
    "EvidenceCheckState",
    "CRCProfile",
    "CRCResult",
    "ParityResult",
    "SyndromeResult",
    "ReencodingResult",
    "FrameRepetitionResult",
    "SyncWordResult",
    "HeaderConsistencyResult",
    "LengthConsistencyResult",
    "StructuralResult",
    "ValidationResult",
    "CRCCalculator",
    "compute_crc",
    "check_crc_frame",
    "search_crc_candidates",
    "load_crc_profiles",
    "reflect_bits",
    "bits_to_bytes",
    "check_even_parity",
    "check_odd_parity",
    "check_block_parity",
    "normalize_syndrome_evidence",
    "estimate_repetition_periods",
    "score_frame_repetition",
    "segment_by_period",
    "search_sync_word",
    "score_sync_periodicity",
    "hex_to_bits",
    "validate_constant_field",
    "validate_monotonic_counter",
    "validate_header_schema",
    "validate_length_field",
    "extract_structural_evidence",
    "reencode_and_compare",
    "evaluate_evidence_groups",
    "compute_overall_validation_score",
    "determine_validation_status",
    "extract_stage11_features",
    "VALIDATION_FEATURE_SCHEMA_VERSION",
    "ValidationEngine",
    "generate_synthetic_frame",
    "generate_synthetic_stream",
    "compute_bitstream_hash",
]
