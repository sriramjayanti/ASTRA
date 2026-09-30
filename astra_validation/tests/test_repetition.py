"""
test_repetition.py
Unit tests for frame repetition and periodicity estimation.
"""

import pytest
import numpy as np
from astra_validation.src.repetition import estimate_repetition_periods, score_frame_repetition
from astra_validation.src.models import EvidenceCheckState


def test_frame_periodicity_detected():
    # Construct 6 frames of 128 bits with a fixed 32-bit header and changing random payload
    np.random.seed(42)
    header = np.random.randint(0, 2, size=32, dtype=np.uint8)
    frames = []
    for _ in range(6):
        payload = np.random.randint(0, 2, size=96, dtype=np.uint8)
        frames.append(np.concatenate([header, payload]))

    stream = np.concatenate(frames)
    res = score_frame_repetition(stream, min_period=32, max_period=256)

    assert res.candidate_period_bits == 128
    assert res.repetition_count >= 5
    assert res.periodicity_score > 0.15
    assert res.check_state == EvidenceCheckState.PASS


def test_random_stream_no_periodicity():
    # Random unperiodic stream
    np.random.seed(999)
    stream = np.random.randint(0, 2, size=1024, dtype=np.uint8)
    res = score_frame_repetition(stream, min_period=32, max_period=256)
    assert res.periodicity_score < 0.20
