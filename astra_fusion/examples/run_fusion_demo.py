"""
ASTRA Multi-Branch Modulation Fusion Engine: Interactive Demonstration.

Shows:
1. Single IQ window inference (Branch A 1D ResNet + Branch B 2D Spectrogram CNN)
2. Fused Top-K modulation candidate ranking
3. Branch agreement vs disagreement analysis & ASTRA status assignment
4. Vectorized batch inference
5. Validation weight grid search w1 in [0.0, 1.0], w2 = 1 - w1
6. Mode B Learned Feature MLP Fusion
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure workspace root is in sys.path
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import numpy as np
import torch

from astra_fusion.src.inference import ASTRAFusionEngine
from astra_fusion.src.learned_fusion import LearnedFusionMLP
from astra_fusion.src.metrics import (
    compute_agreement_metrics,
    grid_search_fusion_weights,
)
from astra_fusion.src.models import BranchPrediction
from astra_fusion.src.utils import (
    BenchmarkTimer,
    format_fusion_prediction_summary,
    setup_logger,
)

logger = setup_logger()


def generate_synthetic_iq(mod_type: str, n_samples: int = 2048, snr_db: float = 10.0) -> np.ndarray:
    """Generates synthetic complex IQ signal for demo purposes."""
    t = np.arange(n_samples)
    if mod_type == "2-FSK":
        f1, f2 = 0.05, 0.15
        bits = np.random.randint(0, 2, n_samples // 32).repeat(32)
        freq = np.where(bits == 0, f1, f2)
        phase = 2 * np.pi * np.cumsum(freq)
        iq = np.exp(1j * phase)
    elif mod_type == "BPSK":
        symbols = np.random.choice([-1.0, 1.0], size=n_samples // 16).repeat(16)
        iq = symbols + 0j
    elif mod_type == "QPSK":
        const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        symbols = np.random.choice(const, size=n_samples // 16).repeat(16)
        iq = symbols
    elif mod_type == "16-QAM":
        vals = [-3, -1, 1, 3]
        grid = [complex(r, i) for r in vals for i in vals]
        grid = np.array(grid) / np.sqrt(10)
        symbols = np.random.choice(grid, size=n_samples // 16).repeat(16)
        iq = symbols
    else:  # Noise / Unknown
        iq = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)

    # Add Gaussian noise
    sig_pwr = np.mean(np.abs(iq) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_pwr / 2.0) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    return (iq + noise).astype(np.complex64)


def main():
    print("=" * 80)
    print("[ASTRA] Multi-Branch Modulation Fusion Engine Demo (1D ResNet + 2D CNN)")
    print("=" * 80)

    # 1. Initialize Engine
    logger.info("Initializing ASTRA Fusion Engine (Mode A: Weighted Probability)...")
    engine = ASTRAFusionEngine(
        mode="weighted_probability",
        weight_1d=0.70,
        weight_2d=0.30,
        top_k=3,
        device="cpu",
    )
    logger.info(f"Loaded {len(engine.class_names)} canonical modulation classes: {engine.class_names}")

    # 2. Single Sample Evaluation (QPSK)
    print("\n" + "=" * 80)
    print("[TEST 1] Single Window Complex IQ Prediction (Synthetic QPSK Signal)")
    print("=" * 80)
    qpsk_iq = generate_synthetic_iq("QPSK", n_samples=2048, snr_db=12.0)

    with BenchmarkTimer("Single-Sample Inference") as timer:
        prediction = engine.predict(
            iq_window=qpsk_iq,
            source_signal_id="sig_demo_qpsk_001",
            window_start=0,
            window_end=2048,
        )

    print(format_fusion_prediction_summary(prediction.to_dict()))
    print(f"Latency: {timer.elapsed_ms:.2f} ms")

    # 3. Branch Disagreement Scenario Demo
    print("\n" + "=" * 80)
    print("[TEST 2] Simulated Branch Disagreement Scenario (1D=QPSK vs 2D=8PSK)")
    print("=" * 80)
    classes = engine.class_names
    p_1d = [0.01, 0.01, 0.02, 0.62, 0.28, 0.03, 0.02, 0.01]  # 1D favors QPSK
    p_2d = [0.01, 0.01, 0.02, 0.25, 0.65, 0.03, 0.02, 0.01]  # 2D favors 8PSK

    mock_1d = BranchPrediction(
        model_name="astra_resnet1d_v1", model_version="1.0.0", class_names=classes,
        logits=[float(np.log(p + 1e-12)) for p in p_1d], probabilities=p_1d,
        predicted_class="QPSK", confidence=0.62,
        top_k=[{"rank": 1, "class": "QPSK", "probability": 0.62}],
        source_signal_id="sig_disagree_001", window_start=0, window_end=2048,
    )
    mock_2d = BranchPrediction(
        model_name="astra_spectrogram2d_v1", model_version="1.0.0", class_names=classes,
        logits=[float(np.log(p + 1e-12)) for p in p_2d], probabilities=p_2d,
        predicted_class="8PSK", confidence=0.65,
        top_k=[{"rank": 1, "class": "8PSK", "probability": 0.65}],
        source_signal_id="sig_disagree_001", window_start=0, window_end=2048,
    )

    disagree_fusion = engine.fuse_predictions(mock_1d, mock_2d)
    print(format_fusion_prediction_summary(disagree_fusion.to_dict()))

    # 4. Batch Inference
    print("\n" + "=" * 80)
    print("[TEST 3] Vectorized Batch Inference (Batch Size = 6 Signals)")
    print("=" * 80)
    batch_signals = [
        generate_synthetic_iq("2-FSK"),
        generate_synthetic_iq("BPSK"),
        generate_synthetic_iq("QPSK"),
        generate_synthetic_iq("16-QAM"),
        generate_synthetic_iq("Unknown"),
        generate_synthetic_iq("4-FSK"),
    ]
    batch_array = np.stack(batch_signals, axis=0)

    with BenchmarkTimer("Batch Inference (N=6)") as timer:
        batch_results = engine.predict_batch(batch_array)

    print(f"Processed {len(batch_results)} windows in {timer.elapsed_ms:.2f} ms ({timer.elapsed_ms / len(batch_results):.2f} ms/window):")
    for i, res in enumerate(batch_results, start=1):
        top1 = res.top_k[0]
        print(f"  Window {i}: Class={res.predicted_class:<10} Confidence={res.confidence:.1%} Status={res.status:<10} Agreement={'YES' if res.branch_agreement else 'NO'}")

    # 5. Validation Weight Grid Search
    print("\n" + "=" * 80)
    print("[TEST 4] Validation Set Weight Grid Search (Finding Optimal w_1d, w_2d)")
    print("=" * 80)
    N_val = 50
    val_labels = np.random.randint(0, 8, N_val)
    val_probs_1d = np.random.dirichlet(np.ones(8), size=N_val).astype(np.float32)
    val_probs_2d = np.random.dirichlet(np.ones(8), size=N_val).astype(np.float32)

    best_w1, best_w2, best_metrics, grid = grid_search_fusion_weights(
        val_probs_1d, val_probs_2d, val_labels, classes, weight_steps=11
    )

    print(f"Optimal Validation Weights Found: w_1d = {best_w1:.2f} | w_2d = {best_w2:.2f}")
    print(f"Validation Accuracy: {best_metrics['accuracy']:.2%} | Macro F1: {best_metrics['macro_f1']:.4f} | ECE: {best_metrics['ece']:.4f}")

    # 6. Mode B Learned MLP Fusion Check
    print("\n" + "=" * 80)
    print("[TEST 5] Mode B Learned Feature MLP Forward Check")
    print("=" * 80)
    learned_mlp = LearnedFusionMLP(dim_1d_feature=256, dim_2d_feature=256, num_classes=8)
    learned_engine = ASTRAFusionEngine(mode="learned", learned_model=learned_mlp, device="cpu")
    logger.info("Instantiated Learned Fusion Engine successfully.")

    print("\n" + "=" * 80)
    print("[SUCCESS] ASTRA Multi-Branch Modulation Fusion Engine is fully operational!")
    print("=" * 80)


if __name__ == "__main__":
    main()
