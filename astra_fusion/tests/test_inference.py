"""
Test suite for ASTRA Fusion Engine End-to-End Inference.
"""

import numpy as np
import pytest
import torch
from astra_fusion.src.inference import ASTRAFusionEngine


def test_14_single_sample_inference():
    """TEST 14: End-to-end single complex IQ sample prediction through ASTRAFusionEngine."""
    engine = ASTRAFusionEngine(device="cpu")
    iq_signal = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
    
    pred = engine.predict(iq_signal, source_signal_id="sig_test_01", window_start=0, window_end=2048)
    assert pred.predicted_class in engine.class_names
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_k) == 3
    assert pred.status in ["CONFIRMED", "ESTIMATED", "POSSIBLE", "UNKNOWN"]
    assert "resnet1d" in pred.branch_evidence
    assert "spectrogram2d" in pred.branch_evidence


def test_15_batch_inference():
    """TEST 15: End-to-end batch inference across multiple IQ windows."""
    engine = ASTRAFusionEngine(device="cpu")
    batch_iq = (np.random.randn(5, 2048) + 1j * np.random.randn(5, 2048)).astype(np.complex64)
    
    results = engine.predict_batch(batch_iq)
    assert len(results) == 5
    for r in results:
        assert r.predicted_class in engine.class_names
        assert 0.0 <= r.confidence <= 1.0
        assert len(r.top_k) == 3


def test_16_cpu_inference():
    """TEST 16: Explicit CPU device inference execution."""
    engine = ASTRAFusionEngine(device="cpu")
    iq_signal = np.random.randn(2, 2048).astype(np.float32)
    pred = engine.predict(iq_signal)
    assert pred is not None
    assert np.isfinite(pred.confidence)


def test_17_cuda_inference_if_available():
    """TEST 17: CUDA inference execution if a GPU is accessible."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = ASTRAFusionEngine(device=device)
    iq_signal = np.random.randn(2, 2048).astype(np.float32)
    pred = engine.predict(iq_signal)
    assert pred is not None


def test_18_nan_input_handling():
    """TEST 18: IQ signals with NaNs or Infs are sanitized safely without crashing."""
    engine = ASTRAFusionEngine(device="cpu")
    corrupt_signal = np.random.randn(2048).astype(np.float32)
    corrupt_signal[100:150] = np.nan
    corrupt_signal[200:220] = np.inf

    pred = engine.predict(corrupt_signal)
    assert pred is not None
    assert np.isfinite(pred.confidence)
    assert np.isfinite(pred.probability_entropy)
