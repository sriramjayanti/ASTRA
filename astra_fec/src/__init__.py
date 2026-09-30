"""
__init__.py
ASTRA Stage 9 — FEC Candidate Testing Engine.
"""

from .models import (
    FECStatus,
    FECFamily,
    FECProfile,
    DecoderResult,
    FECCandidateResult,
    FECTestResult,
)

from .profiles import (
    ProfileRegistry,
    get_profile_registry,
)

from .no_fec import decode_no_fec

from .convolutional import (
    ConvolutionalTrellis,
    encode_convolutional,
    depuncture_stream,
)

from .viterbi import (
    viterbi_decode_hard,
    viterbi_decode_soft,
    decode_convolutional_profile,
)

from .reed_solomon import (
    GaloisField,
    ReedSolomonCodec,
    bits_to_symbols,
    symbols_to_bits,
    decode_reed_solomon_profile,
)

from .ldpc import (
    construct_qc_ldpc_matrix,
    LDPCCodec,
    decode_ldpc_profile,
)

from .concatenated import decode_concatenated_profile

from .candidate_generator import (
    FECCandidateGenerator,
    FECHypothesis,
)

from .metrics import normalize_decoder_metrics
from .scoring import FECScorer
from .pruning import prune_fec_candidates
from .validators import validate_fec_config, validate_interleaver_input
from .router import execute_decoder
from .inference import FECTestingEngine
from .utils import (
    generate_synthetic_fec_stream,
    compute_ber,
    evaluate_fec_candidate_recall,
    format_fec_summary,
)

__all__ = [
    "FECStatus",
    "FECFamily",
    "FECProfile",
    "DecoderResult",
    "FECCandidateResult",
    "FECTestResult",
    "ProfileRegistry",
    "get_profile_registry",
    "decode_no_fec",
    "ConvolutionalTrellis",
    "encode_convolutional",
    "depuncture_stream",
    "viterbi_decode_hard",
    "viterbi_decode_soft",
    "decode_convolutional_profile",
    "GaloisField",
    "ReedSolomonCodec",
    "bits_to_symbols",
    "symbols_to_bits",
    "decode_reed_solomon_profile",
    "construct_qc_ldpc_matrix",
    "LDPCCodec",
    "decode_ldpc_profile",
    "decode_concatenated_profile",
    "FECCandidateGenerator",
    "FECHypothesis",
    "normalize_decoder_metrics",
    "FECScorer",
    "prune_fec_candidates",
    "validate_fec_config",
    "validate_interleaver_input",
    "execute_decoder",
    "FECTestingEngine",
    "generate_synthetic_fec_stream",
    "compute_ber",
    "evaluate_fec_candidate_recall",
    "format_fec_summary",
]
