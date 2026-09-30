"""
astra_bitstream_intelligence
ASTRA Stage 12 — Bitstream Intelligence Engine
"""

from .models import (
    StructureStatus,
    AutocorrPeak,
    PeriodicityCandidate,
    SyncPatternResult,
    RepeatedPattern,
    RunLengthStats,
    BitBalance,
    ByteAlignmentResult,
    StructuralRegion,
    BitstreamIntelligenceResult
)
from .validation import validate_bitstream
from .entropy import (
    binary_entropy,
    block_entropy,
    compute_all_ngram_entropies,
    sliding_entropy,
    detect_entropy_change_points
)
from .autocorrelation import (
    bits_to_bipolar,
    direct_autocorrelation,
    fft_autocorrelation,
    bit_autocorrelation,
    find_autocorrelation_peaks
)
from .cross_correlation import (
    hex_to_bits,
    search_known_sync
)
from .periodicity import (
    is_harmonic,
    merge_harmonics,
    rank_period_candidates
)
from .frame_length import estimate_frame_lengths
from .repeated_patterns import (
    find_repeated_patterns,
    analyze_repeated_prefixes
)
from .run_length import (
    compute_run_lengths,
    calculate_run_length_stats
)
from .bit_balance import calculate_bit_balance
from .byte_alignment import (
    bits_to_bytes,
    compute_byte_entropy,
    analyze_byte_offsets
)
from .segmentation import (
    build_frame_matrix,
    compute_position_stability,
    compute_position_entropy,
    compute_xor_frame_differences,
    compute_frame_hamming_distances,
    find_optimal_frame_offset,
    detect_structural_regions
)
from .sync_candidates import discover_candidate_sync
from .feature_builder import BitstreamFeatureBuilder, FEATURE_SCHEMA_VERSION
from .inference import BitstreamIntelligenceEngine
from .utils import (
    compute_bitstream_hash,
    generate_synthetic_framed_bitstream
)

__all__ = [
    "StructureStatus",
    "AutocorrPeak",
    "PeriodicityCandidate",
    "SyncPatternResult",
    "RepeatedPattern",
    "RunLengthStats",
    "BitBalance",
    "ByteAlignmentResult",
    "StructuralRegion",
    "BitstreamIntelligenceResult",
    "validate_bitstream",
    "binary_entropy",
    "block_entropy",
    "compute_all_ngram_entropies",
    "sliding_entropy",
    "detect_entropy_change_points",
    "bits_to_bipolar",
    "direct_autocorrelation",
    "fft_autocorrelation",
    "bit_autocorrelation",
    "find_autocorrelation_peaks",
    "hex_to_bits",
    "search_known_sync",
    "is_harmonic",
    "merge_harmonics",
    "rank_period_candidates",
    "estimate_frame_lengths",
    "find_repeated_patterns",
    "analyze_repeated_prefixes",
    "compute_run_lengths",
    "calculate_run_length_stats",
    "calculate_bit_balance",
    "bits_to_bytes",
    "compute_byte_entropy",
    "analyze_byte_offsets",
    "build_frame_matrix",
    "compute_position_stability",
    "compute_position_entropy",
    "compute_xor_frame_differences",
    "compute_frame_hamming_distances",
    "find_optimal_frame_offset",
    "detect_structural_regions",
    "discover_candidate_sync",
    "BitstreamFeatureBuilder",
    "FEATURE_SCHEMA_VERSION",
    "BitstreamIntelligenceEngine",
    "compute_bitstream_hash",
    "generate_synthetic_framed_bitstream"
]
