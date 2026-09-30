"""
run_length.py
Run-length distribution and statistics analysis for 0s and 1s.
"""

from typing import List
import numpy as np
from .models import RunLengthStats


def compute_run_lengths(bits: np.ndarray) -> List[int]:
    """Extract consecutive run lengths from 1D binary bitstream."""
    if len(bits) == 0:
        return []

    # Find indices where bit changes
    changes = np.diff(bits) != 0
    change_indices = np.where(changes)[0]

    # Prepend 0 and append length
    run_starts = np.concatenate(([0], change_indices + 1))
    run_ends = np.concatenate((change_indices + 1, [len(bits)]))

    run_lengths = (run_ends - run_starts).tolist()
    return run_lengths


def calculate_run_length_stats(
    bits: np.ndarray,
    long_run_threshold: int = 8
) -> RunLengthStats:
    """
    Compute comprehensive run-length statistics.

    Args:
        bits: 1D uint8 array of binary bits.
        long_run_threshold: Run length considered a 'long run'.

    Returns:
        RunLengthStats dataclass instance.
    """
    if bits is None or len(bits) == 0:
        return RunLengthStats()

    runs = compute_run_lengths(bits)
    if not runs:
        return RunLengthStats()

    arr_runs = np.array(runs, dtype=np.float32)

    mean_rl = float(np.mean(arr_runs))
    median_rl = float(np.median(arr_runs))
    max_rl = int(np.max(arr_runs))
    var_rl = float(np.var(arr_runs))

    long_runs_count = np.sum(arr_runs >= long_run_threshold)
    long_run_frac = float(long_runs_count / len(arr_runs))

    return RunLengthStats(
        mean_run_length=mean_rl,
        median_run_length=median_rl,
        max_run_length=max_rl,
        run_length_variance=var_rl,
        long_run_fraction=long_run_frac
    )
