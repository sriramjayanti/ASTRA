"""
structure.py
Structural telemetry extractor for ASTRA Stage 10 — Validation Engine.
Extracts binary entropy, bit balance, run-length statistics, and byte-alignment quality.
"""

from typing import List, Union
import numpy as np
from .models import StructuralResult


def extract_structural_evidence(bits: Union[np.ndarray, List[int]]) -> StructuralResult:
    """
    Extract structural metrics from a recovered candidate bitstream.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = len(bits)

    if n == 0:
        return StructuralResult()

    # 1. Bit balance (fraction of 0s)
    num_zeros = int(np.sum(bits == 0))
    p0 = num_zeros / n
    p1 = 1.0 - p0

    # 2. Binary Shannon entropy
    if p0 == 0.0 or p1 == 0.0:
        entropy = 0.0
    else:
        entropy = - (p0 * np.log2(p0) + p1 * np.log2(p1))

    # 3. Run length distribution
    if n > 1:
        # Find transition indices
        diffs = np.diff(bits)
        run_starts = np.where(diffs != 0)[0] + 1
        run_indices = np.concatenate(([0], run_starts, [n]))
        run_lengths = np.diff(run_indices)
        mean_run = float(np.mean(run_lengths))
    else:
        mean_run = 1.0

    # 4. Byte alignment quality (checking if 8-bit transitions exhibit structure)
    byte_align_score = 0.0
    if n >= 64:
        usable = (n // 8) * 8
        reshaped = bits[:usable].reshape(-1, 8)
        # Check standard byte MSB / printable distribution variance
        col_means = np.mean(reshaped, axis=0)
        byte_align_score = float(np.std(col_means) * 2.0)  # Elevated std indicates structured byte columns

    return StructuralResult(
        binary_entropy=float(entropy),
        bit_balance_zero=float(p0),
        mean_run_length=float(mean_run),
        byte_alignment_score=float(min(1.0, byte_align_score)),
    )
