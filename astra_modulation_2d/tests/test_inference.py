import numpy as np
import torch

from astra_modulation_2d.src.model import ASTRASpectrogramCNN
from astra_modulation_2d.src.inference import ASTRASpectrogramPredictor


def test_10_probabilities_sum_to_one():
    """TEST 10: Inference probabilities vector sums to 1.0."""
    classes = ["2fsk", "4fsk", "bpsk", "qpsk", "8psk", "dqpsk", "msk", "16qam", "64qam", "256qam"]
    model = ASTRASpectrogramCNN(class_names=classes)
    predictor = ASTRASpectrogramPredictor(model, top_k=3)

    iq = np.random.randn(2048) + 1j * np.random.randn(2048)
    res = predictor.predict(iq)

    prob_sum = sum(res["probabilities"].values())
    assert np.isclose(prob_sum, 1.0, atol=1e-4), f"Probabilities sum was {prob_sum}, expected 1.0"


def test_11_top_k_sorted():
    """TEST 11: Top-K candidate list is strictly sorted in descending probability order."""
    classes = ["bpsk", "qpsk", "8psk", "16qam", "64qam"]
    model = ASTRASpectrogramCNN(class_names=classes)
    predictor = ASTRASpectrogramPredictor(model, top_k=4)

    iq = np.random.randn(2048) + 1j * np.random.randn(2048)
    res = predictor.predict(iq)

    top_k = res["top_k"]
    assert len(top_k) == 4
    for i in range(len(top_k) - 1):
        assert top_k[i]["probability"] >= top_k[i + 1]["probability"]


def test_12_cpu_inference():
    """TEST 12: Predictor operates reliably on CPU device."""
    classes = ["bpsk", "qpsk"]
    model = ASTRASpectrogramCNN(class_names=classes)
    predictor = ASTRASpectrogramPredictor(model, device=torch.device("cpu"))

    iq = np.random.randn(2048) + 1j * np.random.randn(2048)
    res = predictor.predict(iq)
    assert res["predicted_class"] in classes
    assert 0.0 <= res["confidence"] <= 1.0


def test_13_cuda_inference_if_available():
    """TEST 13: Predictor operates on CUDA GPU if CUDA is available."""
    if not torch.cuda.is_available():
        return

    classes = ["bpsk", "qpsk"]
    model = ASTRASpectrogramCNN(class_names=classes)
    predictor = ASTRASpectrogramPredictor(model, device=torch.device("cuda"))

    iq = np.random.randn(2048) + 1j * np.random.randn(2048)
    res = predictor.predict(iq)
    assert res["predicted_class"] in classes


def test_18_batch_inference():
    """TEST 18: Batch inference evaluates multiple windows consistently."""
    classes = ["bpsk", "qpsk", "8psk", "16qam"]
    model = ASTRASpectrogramCNN(class_names=classes)
    predictor = ASTRASpectrogramPredictor(model)

    batch_iq = [np.random.randn(2048) + 1j * np.random.randn(2048) for _ in range(5)]
    results = predictor.predict_batch(batch_iq)
    assert len(results) == 5
    for r in results:
        assert r["predicted_class"] in classes
        assert "probabilities" in r
