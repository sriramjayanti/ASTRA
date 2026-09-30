"""
__init__.py
ASTRA Stage 8 — Interleaver Candidate Testing Engine.
"""

from .models import (
    InterleaverStatus,
    InterleaverFamily,
    PermutationMapping,
    ConvolutionalDeinterleaverState,
    StructuralFeatures,
    InterleaverCandidateResult,
    InterleaverTestResult,
)

from .permutation import (
    invert_permutation,
    validate_permutation,
    compute_permutation_hash,
    apply_inverse_mapping,
    apply_block_chunked_inverse,
    compute_permutation_distance,
)

from .identity import (
    create_identity_mapping,
    deinterleave_identity,
    interleave_identity,
)

from .block import (
    create_block_mapping,
    interleave_block,
    deinterleave_block,
)

from .convolutional import (
    ConvolutionalInterleaverEngine,
    ConvolutionalDeinterleaverEngine,
    interleave_convolutional,
    deinterleave_convolutional,
)

from .helical import (
    create_helical_mapping,
    interleave_helical,
    deinterleave_helical,
)

from .pseudo_random import (
    generate_pseudorandom_indices,
    create_pseudorandom_mapping,
    interleave_pseudorandom,
    deinterleave_pseudorandom,
)

from .structural_features import (
    compute_binary_entropy,
    compute_run_length_statistics,
    compute_autocorrelation_features,
    extract_structural_features,
)

from .scoring import CandidateScorer
from .pruning import prune_candidates
from .candidate_generator import InterleaverCandidateGenerator, CandidateHypothesis
from .router import execute_deinterleaver
from .inference import InterleaverTestingEngine
from .validators import validate_config, validate_demod_input
from .utils import (
    generate_synthetic_interleaved_stream,
    evaluate_candidate_recall,
    format_interleaver_summary,
)

__all__ = [
    "InterleaverStatus",
    "InterleaverFamily",
    "PermutationMapping",
    "ConvolutionalDeinterleaverState",
    "StructuralFeatures",
    "InterleaverCandidateResult",
    "InterleaverTestResult",
    "invert_permutation",
    "validate_permutation",
    "compute_permutation_hash",
    "apply_inverse_mapping",
    "apply_block_chunked_inverse",
    "compute_permutation_distance",
    "create_identity_mapping",
    "deinterleave_identity",
    "interleave_identity",
    "create_block_mapping",
    "interleave_block",
    "deinterleave_block",
    "ConvolutionalInterleaverEngine",
    "ConvolutionalDeinterleaverEngine",
    "interleave_convolutional",
    "deinterleave_convolutional",
    "create_helical_mapping",
    "interleave_helical",
    "deinterleave_helical",
    "generate_pseudorandom_indices",
    "create_pseudorandom_mapping",
    "interleave_pseudorandom",
    "deinterleave_pseudorandom",
    "compute_binary_entropy",
    "compute_run_length_statistics",
    "compute_autocorrelation_features",
    "extract_structural_features",
    "CandidateScorer",
    "prune_candidates",
    "InterleaverCandidateGenerator",
    "CandidateHypothesis",
    "execute_deinterleaver",
    "InterleaverTestingEngine",
    "validate_config",
    "validate_demod_input",
    "generate_synthetic_interleaved_stream",
    "evaluate_candidate_recall",
    "format_interleaver_summary",
]
