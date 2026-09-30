# ASTRA Stage 12 — Bitstream Intelligence Engine

Production-ready statistical and deterministic bitstream intelligence engine for ASTRA (Automated Signal Analysis & Recovery Assistant).

Stage 12 receives the best-supported recovered bitstreams from **Stage 11 (Pipeline XGBoost Scorer)** and analyzes their internal structural properties prior to neural sequence modeling in **Stage 13 (1D CNN + Transformer)**.

---

## 1. Core Architecture

```
Stage 11: Pipeline XGBoost Scorer
             ↓
[Best Supported Decoded Bitstream(s)]
             ↓
======================================================
STAGE 12: BITSTREAM INTELLIGENCE ENGINE
  ├─ Binary & Sliding N-Gram Entropy Analysis
  ├─ Direct & FFT Normalized Bipolar Autocorrelation
  ├─ Harmonic Clustering & Fundamental Period Recovery
  ├─ Multi-Evidence Frame-Length Estimation (Top-K)
  ├─ Known Sync Search & Blind Sync Discovery
  ├─ Recurring Bit-Pattern Discovery
  ├─ Positional Stability & Regional Entropy Maps
  ├─ Byte-Alignment Exploration (0..7 bit offsets)
  └─ Stage 13 Multi-Channel Sequence Feature Builder
======================================================
             ↓
Structural Features + Multi-Channel Sequence Map
             ↓
Stage 13: 1D CNN + Transformer (Header/Payload Explorer)
```

---

## 2. Key Capabilities

1. **Bipolar Autocorrelation:** Fast $O(N \log N)$ FFT-based autocorrelation converting bits to $\pm 1$, finding structural peaks while strictly excluding lag 0.
2. **Harmonic Merging:** Clusters peaks at $L, 2L, 3L$ to pinpoint true fundamental frame periods.
3. **Entropy Profile Analysis:** Multi-scale sliding window entropy with change-point detection to identify transitions between sync, header, payload, and padding.
4. **Positional Stability Mapping:** Frame matrix reshaping $[N_{\text{frames}}, L_{\text{frame}}]$ calculating position-wise $P(\text{bit}=1)$, identifying invariant headers and variable payloads.
5. **Byte Alignment Exploration:** Tests all 8 bit offsets ($0 \dots 7$) to check for byte structure without assuming hard alignment.
6. **Multi-Channel Sequence Features for Stage 13:** Emits 4-channel tensors:
   - Channel 0: Bipolar bit value ($\pm 1$)
   - Channel 1: Soft reliability / confidence
   - Channel 2: Local sliding-window entropy
   - Channel 3: Positional stability / frame boundary marker

---

## 3. Directory Structure

```
astra_bitstream_intelligence/
│
├── configs/
│   └── bitstream_intelligence_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── validation.py
│   ├── entropy.py
│   ├── autocorrelation.py
│   ├── cross_correlation.py
│   ├── periodicity.py
│   ├── frame_length.py
│   ├── repeated_patterns.py
│   ├── run_length.py
│   ├── bit_balance.py
│   ├── byte_alignment.py
│   ├── segmentation.py
│   ├── sync_candidates.py
│   ├── feature_builder.py
│   ├── inference.py
│   └── utils.py
│
├── tests/
│   ├── test_entropy.py
│   ├── test_autocorrelation.py
│   ├── test_periodicity.py
│   ├── test_frame_length.py
│   ├── test_repeated_patterns.py
│   ├── test_sync_candidates.py
│   ├── test_byte_alignment.py
│   └── test_end_to_end_bitstream.py
│
├── examples/
│   └── analyze_bitstream.py
│
├── run_tests.py
├── requirements.txt
└── README.md
```

---

## 4. Usage Example

```python
from astra_bitstream_intelligence.src.inference import BitstreamIntelligenceEngine

# Initialize engine
engine = BitstreamIntelligenceEngine()

# Analyze recovered hard bits
result = engine.analyze(
    decoded_bits=recovered_bits,
    context={"pipeline_path_id": "candidate_01"}
)

print(f"Status: {result.status}")
print(f"Top Frame Length: {result.frame_length_candidates[0].period_bits} bits")
print(f"Global Entropy: {result.binary_entropy_global:.4f}")
print(f"Stage 13 Sequence Tensor: {result.sequence_feature_map.shape}")
```

---

## 5. Running Tests

```bash
python astra_bitstream_intelligence/run_tests.py
```
