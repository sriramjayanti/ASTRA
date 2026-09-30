# ASTRA Multi-Branch Modulation Fusion Engine

A production-ready Multi-Branch Modulation Evidence Fusion Engine combining:
1. **Branch A (1D ResNet):** Deep 1D Residual Convolutional Classifier operating on raw normalized complex baseband $I/Q$ waveforms $[B, 2, 2048]$.
2. **Branch B (2D Spectrogram CNN):** Deep 2D Residual Convolutional Classifier with Squeeze-and-Excitation (SE) attention operating on centered baseband STFT spectrograms $[B, 1, 128, 65]$.

---

## 1. Role in the ASTRA End-to-End Pipeline

The Fusion Engine is **NOT** the final ground truth authority. Its mission is to synthesize multi-domain evidence, compute calibrated probabilities, quantify uncertainty, and generate rank-ordered **Top-K Modulation Candidates** to seed the downstream Candidate / Hypothesis Engine:

```
                      Raw Complex Baseband IQ [N = 2048]
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
    ┌──────────────────┐                            ┌──────────────────┐
    │   1D ResNet IQ   │                            │  STFT Generator  │
    │   (Branch A)     │                            │  (Centered PSD)  │
    └─────────┬────────┘                            └─────────┬────────┘
              │                                               │
              │                                               ▼
              │                                     ┌──────────────────┐
              │                                     │  2D ResNet + SE  │
              │                                     │   (Branch B)     │
              │                                     └─────────┬────────┘
              │ (1D Logits / Probs / Embedding)               │ (2D Logits / Probs / Embedding)
              └───────────────────────┬───────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │    ASTRA Fusion Engine    │
                        │  (Mode A / Mode B Fusion) │
                        └─────────────┬─────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
      [Fused Probabilities]  [Top-K Candidates]   [ASTRA Status & Margin]
                │                     │                     │
                └─────────────────────┼─────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Candidate / Hypothesis    │
                        │ Engine (Modulation x Baud)│
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Receiver Demod & Recovery │
                        └───────────────────────────┘
```

---

## 2. Core Fusion Modes

### Mode A: Weighted Probability Fusion (Default & Baseline)
Combines class probabilities via convex linear combination:
$$P_{\text{fused}} = w_{\text{1d}} \cdot P_{\text{1D}} + w_{\text{2d}} \cdot P_{\text{2D}}$$
* Constraints: $w_{\text{1d}} \ge 0$, $w_{\text{2d}} \ge 0$, $w_{\text{1d}} + w_{\text{2d}} = 1.0$.
* Initial Configurable Default: $w_{\text{1d}} = 0.70$, $w_{\text{2d}} = 0.30$.
* Deterministic, zero-parameter overhead, fully explainable.

### Mode B: Learned Feature & Logit Fusion MLP
Synthesizes concatenated dense representations and raw logits:
$$\mathbf{x}_{\text{fuse}} = \left[ \mathbf{z}_{\text{1D}} \,\|\, \mathbf{z}_{\text{2D}} \,\|\, \boldsymbol{\ell}_{\text{1D}} \,\|\, \boldsymbol{\ell}_{\text{2D}} \right] \in \mathbb{R}^{256 + 256 + 8 + 8 = 528}$$
$$\mathbf{h}_1 = \text{Dropout}_{0.2}\left(\text{ReLU}\left(\text{BatchNorm}\left(\mathbf{W}_1 \mathbf{x}_{\text{fuse}}\right)\right)\right) \in \mathbb{R}^{256}$$
$$\mathbf{h}_2 = \text{Dropout}_{0.2}\left(\text{ReLU}\left(\text{BatchNorm}\left(\mathbf{W}_2 \mathbf{h}_1\right)\right)\right) \in \mathbb{R}^{128}$$
$$\boldsymbol{\ell}_{\text{fused}} = \mathbf{W}_3 \mathbf{h}_2 \in \mathbb{R}^{\text{num\_classes}}$$

---

## 3. Strict Verification & Integrity Guarantees

1. **Shared IQ Window Enforcement:** Both branches evaluate the **exact same underlying IQ window**. Fusing predictions from differing source IDs or window ranges raises `SourceAlignmentError`.
2. **Class Alignment Validation:** Validates exact class names, ordering, and length matching between 1D and 2D models (`ClassMappingMismatchError`).
3. **Probability & Convexity Validation:** Checks for NaN, Inf, non-negative bounds, and unit sum (`InvalidProbabilityError`, `InvalidWeightError`).
4. **No Label Leakage:** Fusion depends strictly on model outputs, features, and preprocessed IQ without ground truth labels or metadata cues.

---

## 4. Confidence Metrics, Entropy & ASTRA Status Tiers

* **Top-1 Confidence:** $c = \max(P_{\text{fused}})$.
* **Confidence Margin:** $\Delta = P_{\text{fused}}^{(\text{top 1})} - P_{\text{fused}}^{(\text{top 2})}$.
* **Normalized Shannon Entropy:**
  $$H_{\text{norm}}(P) = -\frac{1}{\log_2(C)} \sum_{i=1}^C P_i \log_2(P_i + \epsilon) \in [0.0, 1.0]$$

### ASTRA Status Tier Decision Matrix

| Status Tier | Conditions | Meaning |
| :--- | :--- | :--- |
| **`CONFIRMED`** | Agreement $\land$ $c \ge 0.85$ $\land$ $\Delta \ge 0.20$ | High certainty consensus between time and frequency domains |
| **`ESTIMATED`** | $c \ge 0.50$ | Strong candidate hypothesis ready for demodulation testing |
| **`POSSIBLE`** | $c \ge 0.30$ | Moderate evidence candidate; downstream decoder check required |
| **`UNKNOWN`** | $c < 0.30$ $\lor$ Explicit "Unknown" Class $\lor$ High Conflict | Ambiguous signal; tagged with precise `unknown_reason` |

#### Unknown Reasons Tracked
* `"explicit_unknown_class"`: Model identified noise / out-of-distribution class.
* `"low_confidence"`: Winner probability below acceptance threshold.
* `"high_uncertainty"`: High distribution entropy across multiple classes.
* `"branch_conflict"`: 1D and 2D branches disagree with low margin.

---

## 5. Standard Output JSON Schema

```json
{
  "predicted_class": "QPSK",
  "confidence": 0.88,
  "status": "CONFIRMED",
  "unknown_reason": null,
  "branch_agreement": true,
  "confidence_margin": 0.74,
  "probability_entropy": 0.21,
  "fusion_mode": "weighted_probability",
  "fusion_weights": {
    "resnet1d": 0.70,
    "spectrogram2d": 0.30
  },
  "top_k": [
    {
      "rank": 1,
      "class": "QPSK",
      "probability": 0.88
    },
    {
      "rank": 2,
      "class": "8PSK",
      "probability": 0.14
    },
    {
      "rank": 3,
      "class": "16-QAM",
      "probability": 0.03
    }
  ],
  "branch_evidence": {
    "resnet1d": {
      "model_name": "astra_resnet1d_v1",
      "predicted_class": "QPSK",
      "confidence": 0.92
    },
    "spectrogram2d": {
      "model_name": "astra_spectrogram2d_v1",
      "predicted_class": "QPSK",
      "confidence": 0.79
    }
  },
  "model_versions": {
    "resnet1d": "1.0.0",
    "spectrogram2d": "1.0.0",
    "fusion": "astra_weighted_fusion_v1.0"
  },
  "source_signal_id": "sig_demo_001",
  "window_start": 0,
  "window_end": 2048
}
```

---

## 6. Python API Usage

### Single Sample Inference
```python
from astra_fusion import ASTRAFusionEngine
import numpy as np

# Instantiate engine
engine = ASTRAFusionEngine(
    config_path="astra_fusion/configs/fusion_config.yaml",
    device="cpu"
)

# Complex IQ window [N = 2048]
iq_signal = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)

# Predict
result = engine.predict(
    iq_window=iq_signal,
    source_signal_id="sig_001",
    window_start=0,
    window_end=2048
)

print(f"Predicted: {result.predicted_class} ({result.confidence:.1%}) | Status: {result.status}")
print("Top-K Candidates:", result.top_k)
```

### Vectorized Batch Inference
```python
# Batch of IQ windows [B = 32, N = 2048]
batch_iq = (np.random.randn(32, 2048) + 1j * np.random.randn(32, 2048)).astype(np.complex64)
batch_results = engine.predict_batch(batch_iq)
```

### Validation Weight Search Utility
```python
from astra_fusion import grid_search_fusion_weights

# Grid search w_1d in [0.0, 1.0], w_2d = 1 - w_1d strictly on validation split
best_w1, best_w2, best_metrics, grid_log = grid_search_fusion_weights(
    probs_1d=val_probs_1d,
    probs_2d=val_probs_2d,
    y_true=val_labels,
    class_names=engine.class_names,
    weight_steps=11
)
print(f"Optimal weights: w_1d={best_w1:.2f}, w_2d={best_w2:.2f} (Macro F1={best_metrics['macro_f1']:.4f})")
```

---

## 7. Unit & Integration Test Suite

The test suite covers all 20 required criteria:

```bash
python astra_fusion/run_tests.py
```

### Verified Test Matrix
* **TEST 1:** Class mapping equality check (`test_1_class_mapping_equality`)
* **TEST 2:** Mismatched mapping exception (`test_2_mismatched_mapping_raises_error`)
* **TEST 3:** Source alignment validation (`test_3_source_alignment_works`)
* **TEST 4:** Unrelated source window rejection (`test_4_unrelated_source_windows_rejected`)
* **TEST 5:** Weighted probabilities sum to 1.0 (`test_5_weighted_probabilities_sum_to_1`)
* **TEST 6:** Weight convex sum validation (`test_6_weights_sum_to_1`)
* **TEST 7:** Top-K candidate sorting (`test_7_top_k_sorted`)
* **TEST 8:** Branch agreement detection (`test_8_branch_agreement`)
* **TEST 9:** Branch disagreement tracking (`test_9_branch_disagreement`)
* **TEST 10:** Confidence margin computation (`test_10_confidence_margin`)
* **TEST 11:** Normalized Shannon entropy bounds (`test_11_entropy_finite`)
* **TEST 12:** Low confidence UNKNOWN status (`test_12_unknown_status`)
* **TEST 13:** Explicit Unknown class handling (`test_13_explicit_unknown_class`)
* **TEST 14:** Single-sample end-to-end inference (`test_14_single_sample_inference`)
* **TEST 15:** Vectorized batch inference (`test_15_batch_inference`)
* **TEST 16:** Explicit CPU execution (`test_16_cpu_inference`)
* **TEST 17:** CUDA GPU execution if available (`test_17_cuda_inference_if_available`)
* **TEST 18:** NaN and Inf input sanitization (`test_18_nan_input_handling`)
* **TEST 19:** Mode B checkpoint save & reload fidelity (`test_19_checkpoint_reload`)
* **TEST 20:** Deterministic weighted fusion (`test_20_deterministic_weighted_fusion`)
