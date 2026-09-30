"""
ASTRA Interleaving / Bit-Reordering Ground-Truth Module.
Synthetic Engine 4 for Automated Signal Analysis & Recovery Assistant.
"""

from .models import InterleaverRecord
from .permutation import apply_permutation, invert_permutation, validate_permutation
from .padding import pad_to_multiple, segment_into_blocks, unpad
from .block import BlockInterleaver, compute_block_permutation, interleave_block, deinterleave_block
from .convolutional import ConvolutionalInterleaver, interleave_convolutional, deinterleave_convolutional
from .diagonal import DiagonalInterleaver, compute_diagonal_permutation, interleave_diagonal, deinterleave_diagonal
from .pseudo_random import (
    PseudoRandomInterleaver,
    generate_pseudo_random_permutation,
    interleave_pseudo_random,
    deinterleave_pseudo_random,
)
from .validators import validate_interleaver_record, reference_deinterleave
from .serializers import save_interleaver_record, load_interleaver_record, save_interleaver_batch
from .generator import InterleaverGenerator, BUILTIN_INTERLEAVER_PROFILES

__all__ = [
    "InterleaverGenerator",
    "InterleaverRecord",
    "BlockInterleaver",
    "compute_block_permutation",
    "interleave_block",
    "deinterleave_block",
    "ConvolutionalInterleaver",
    "interleave_convolutional",
    "deinterleave_convolutional",
    "DiagonalInterleaver",
    "compute_diagonal_permutation",
    "interleave_diagonal",
    "deinterleave_diagonal",
    "PseudoRandomInterleaver",
    "generate_pseudo_random_permutation",
    "interleave_pseudo_random",
    "deinterleave_pseudo_random",
    "apply_permutation",
    "invert_permutation",
    "validate_permutation",
    "pad_to_multiple",
    "segment_into_blocks",
    "unpad",
    "validate_interleaver_record",
    "reference_deinterleave",
    "save_interleaver_record",
    "load_interleaver_record",
    "save_interleaver_batch",
    "BUILTIN_INTERLEAVER_PROFILES",
]

