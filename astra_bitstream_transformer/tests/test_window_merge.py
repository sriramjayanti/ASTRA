"""
test_window_merge.py
Unit tests for window slicing, overlap merging, and long-stream inference.
"""

import pytest
import numpy as np
from astra_bitstream_transformer.src.windowing import slice_sequence_windows
from astra_bitstream_transformer.src.merge import merge_overlapping_predictions
from astra_bitstream_transformer.src.inference import BitstreamStructureModel


def test_13_window_slicing():
    """TEST 13: window slicing with padding and overlap."""
    C, N = 6, 2500
    channels = np.random.randn(C, N).astype(np.float32)
    labels = np.random.randint(0, 6, size=N)

    windows, masks, lbl_wins, slices = slice_sequence_windows(
        channels=channels,
        labels=labels,
        window_length=1024,
        overlap_fraction=0.5
    )

    # N=2500, win=1024, step=512 -> windows at [0..1024], [512..1536], [1024..2048], [1536..2500] (padded)
    assert windows.shape[0] == 4
    assert windows.shape[1] == 6
    assert windows.shape[2] == 1024
    assert masks.shape == (4, 1024)
    assert lbl_wins.shape == (4, 1024)
    # The last window should have padding
    assert np.any(masks[-1])


def test_14_overlap_merging():
    """TEST 14: overlapping window probability merging."""
    num_windows = 3
    win_len = 100
    num_classes = 6
    total_len = 200

    # Step = 50: [0..100], [50..150], [100..200]
    slices = [(0, 100), (50, 150), (100, 200)]
    win_probs = np.full((num_windows, win_len, num_classes), 1.0 / num_classes, dtype=np.float32)

    full_probs, labels = merge_overlapping_predictions(
        window_probs=win_probs,
        window_slices=slices,
        total_length=total_len,
        num_classes=num_classes,
        window_length=win_len
    )

    assert full_probs.shape == (total_len, num_classes)
    assert labels.shape == (total_len,)
    assert np.allclose(np.sum(full_probs, axis=-1), 1.0, atol=1e-5)


def test_26_long_stream_inference():
    """TEST 26: long bitstream inference (e.g. 10,000 bits) runs smoothly."""
    model = BitstreamStructureModel()
    long_bits = np.random.randint(0, 2, size=10000, dtype=np.uint8)

    pred = model.predict(long_bits)
    assert pred.sequence_length == 10000
    assert len(pred.predicted_labels) == 10000
    assert pred.label_probabilities.shape == (10000, 6)
