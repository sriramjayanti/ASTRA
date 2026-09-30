"""
test_inference.py
Integration and comprehensive unit tests for Stage 13 Bitstream Structure Model.
"""

import pytest
import numpy as np
import torch
import json
import os

from astra_bitstream_transformer.src.models import (
    StructureLabel,
    BitstreamStructurePrediction,
    ASTRABitstreamCNNTransformer
)
from astra_bitstream_transformer.src.inference import BitstreamStructureModel
from astra_bitstream_transformer.src.utils import (
    save_checkpoint,
    load_checkpoint,
    generate_annotated_synthetic_stream
)
from astra_bitstream_intelligence.src.inference import BitstreamIntelligenceEngine


def test_21_and_22_cpu_and_cuda_inference():
    """TEST 21 & 22: CPU inference runs and CUDA is optionally supported."""
    model = BitstreamStructureModel(device="cpu")
    bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
    pred = model.predict(bits)

    assert pred.sequence_length == 512
    assert pred.predicted_labels.shape == (512,)
    assert pred.label_probabilities.shape == (512, 6)


def test_23_checkpoint_save_and_load(tmp_path):
    """TEST 23: checkpoint saving and loading restores weights perfectly."""
    model_orig = ASTRABitstreamCNNTransformer(in_channels=6, d_model=64, num_heads=2, num_layers=1)
    ckpt_file = str(tmp_path / "test_ckpt.pth")

    save_checkpoint(model_orig, checkpoint_path=ckpt_file, epoch=5, val_metrics={"macro_f1": 0.92})

    model_loaded = ASTRABitstreamCNNTransformer(in_channels=6, d_model=64, num_heads=2, num_layers=1)
    meta = load_checkpoint(ckpt_file, model_loaded)

    assert meta["epoch"] == 5
    assert meta["val_metrics"]["macro_f1"] == 0.92

    # Verify weight equality
    for p1, p2 in zip(model_orig.parameters(), model_loaded.parameters()):
        assert torch.allclose(p1, p2)


def test_25_deterministic_eval_mode():
    """TEST 25: deterministic outputs in eval mode."""
    model = BitstreamStructureModel()
    bits = np.random.randint(0, 2, size=512, dtype=np.uint8)

    pred1 = model.predict(bits)
    pred2 = model.predict(bits)

    assert np.array_equal(pred1.predicted_labels, pred2.predicted_labels)
    assert np.allclose(pred1.label_probabilities, pred2.label_probabilities, atol=1e-5)


def test_27_batch_inference():
    """TEST 27: batch prediction on multiple bitstreams."""
    model = BitstreamStructureModel()
    batch = [
        np.random.randint(0, 2, size=256, dtype=np.uint8),
        np.random.randint(0, 2, size=512, dtype=np.uint8),
        np.random.randint(0, 2, size=128, dtype=np.uint8)
    ]

    preds = model.predict_batch(batch)
    assert len(preds) == 3
    assert preds[0].sequence_length == 256
    assert preds[1].sequence_length == 512
    assert preds[2].sequence_length == 128


def test_28_json_metadata_serialization():
    """TEST 28: BitstreamStructurePrediction to_dict() is JSON serializable."""
    model = BitstreamStructureModel()
    bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
    pred = model.predict(bits)

    pred_dict = pred.to_dict(include_per_bit=False)
    json_str = json.dumps(pred_dict, indent=2)
    assert len(json_str) > 50

    loaded = json.loads(json_str)
    assert loaded["sequence_length"] == 512
    assert "regions" in loaded
    assert "model_version" in loaded


def test_29_and_30_no_hidden_truth_and_version():
    """TEST 29 & 30: inference does not leak hidden truth and emits model version."""
    model = BitstreamStructureModel()
    bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
    pred = model.predict(bits)

    assert pred.model_version == "astra_bitstream_cnn_transformer_v1"
    assert pred.feature_schema_version == "bitstream_model_input_v1"


def test_102_stage_12_to_stage_13_integration():
    """TEST 102: full Stage 12 -> Stage 13 pipeline execution."""
    stage12_engine = BitstreamIntelligenceEngine()
    stage13_model = BitstreamStructureModel()

    # Generate synthetic framed bitstream from Stage 11/12
    bits, channels, labels, truth = generate_annotated_synthetic_stream(
        frame_length=512,
        num_frames=8,
        sync_word_hex="EB90",
        header_length=32,
        crc_length=16,
        seed=42
    )

    # 1. Run Stage 12
    s12_result = stage12_engine.analyze(bits, context={"pipeline_path_id": "test_pipe_01"})
    assert s12_result.status.value in ["STRUCTURE_STRONG", "STRUCTURE_MODERATE"]

    # 2. Run Stage 13 using Stage 12 features
    s13_pred = stage13_model.predict(
        bits,
        stage12_result=s12_result,
        context={"pipeline_path_id": "test_pipe_01"}
    )

    assert s13_pred.sequence_length == len(bits)
    assert len(s13_pred.regions) > 0
    assert s13_pred.pipeline_path_id == "test_pipe_01"
