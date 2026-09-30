"""
test_dataset.py
Unit tests for Stage 12 feature alignment, missing channel handling, and soft confidence channels.
"""

import pytest
import numpy as np
import torch
from astra_bitstream_transformer.src.dataset import create_synthetic_dataset
from astra_bitstream_transformer.src.inference import BitstreamStructureModel


def test_18_stage_12_feature_alignment():
    """TEST 18: Stage 12 feature alignment ensures each channel matches exact bit indices."""
    model = BitstreamStructureModel()
    bits = np.random.randint(0, 2, size=512, dtype=np.uint8)

    channels = model._prepare_channel_tensor(bits)
    assert channels.shape == (6, 512)
    # Channel 0 should be exact bipolar conversion of bits
    assert np.allclose(channels[0, :], 2.0 * bits - 1.0)


def test_19_missing_feature_channels_handled():
    """TEST 19: missing Stage 12 features default gracefully without crashing."""
    model = BitstreamStructureModel()
    bits = np.random.randint(0, 2, size=300, dtype=np.uint8)

    # Calling predict without Stage 12 results
    pred = model.predict(bits)
    assert pred.sequence_length == 300
    assert len(pred.predicted_labels) == 300


def test_20_bit_confidence_channel():
    """TEST 20: bit confidence channel integration."""
    model = BitstreamStructureModel()
    bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
    soft = np.random.uniform(0.5, 4.0, size=256).astype(np.float32)

    channels = model._prepare_channel_tensor(bits, soft_info=soft)
    assert np.all(channels[1, :] >= 0.0)
    assert np.all(channels[1, :] <= 1.0)
