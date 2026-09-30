# ASTRA Stage 13 — 1D CNN + Transformer Bitstream Structure Model

Production-ready neural sequence model for ASTRA (Automated Signal Analysis & Recovery Assistant).

Stage 13 receives recovered post-FEC bitstreams and Stage 12 structural telemetry to predict higher-level bitstream regions prior to content parsing in **Stage 14 (Header / Payload Explorer)**.

---

## 1. Architecture Flow

```
Stage 12: Bitstream Intelligence Engine
  (Raw Bits, Local Entropy, Periodicity, Sync Impulses, Positional Stability)
             ↓
=============================================================================
STAGE 13: 1D CNN + TRANSFORMER BITSTREAM STRUCTURE MODEL
  ├─ 6-Channel Input Sequence Tensor [B, 6, L] (bit-level aligned)
  │    Ch 0: Bipolar Bit Value (-1.0 / +1.0)
  │    Ch 1: Soft Reliability / Confidence (0.0 to 1.0)
  │    Ch 2: Local Sliding Window Entropy (0.0 to 1.0)
  │    Ch 3: Frame Boundary / Periodic Ramp Score (0.0 to 1.0)
  │    Ch 4: Sync Candidate Evidence / Impulse (0.0 to 1.0)
  │    Ch 5: Positional Stability Map (0.0 to 1.0)
  │
  ├─ 1D CNN Local Motif Extractor (Kernel=7, Stride=1, Residual Blocks)
  │    Preserves exact bit-level sequence resolution without downsampling
  │
  ├─ Sinusoidal Positional Encoding [B, L, 256]
  │
  ├─ Transformer Encoder Stack (4 layers, 8 heads, d_model=256, dim_ff=1024)
  │    Captures long-range dependencies, counters, headers, and framing context
  │
  ├─ Multi-Task Prediction Heads:
  │    1. Per-Position Sequence Head -> [B, L, 6] (UNKNOWN, SYNC, HEADER, PAYLOAD, CRC, PADDING)
  │    2. Auxiliary Boundary Head    -> [B, L] (1.0 at region transitions)
  │    3. Masked Frame Head          -> [B, 4] (valid_structure, header, payload, crc)
  │
  └─ Windowing & Inference Postprocessing:
       - 2048-bit overlapping windowing with triangular/Hann prediction merging
       - Contiguous region parsing with spike filtering and uncertainty estimation
=============================================================================
             ↓
Parsed Bitstream Structure (`BitstreamStructurePrediction`)
             ↓
Stage 14: Header / Payload Explorer
```

---

## 2. Target Structure Labels

| ID | Label | Description |
| :---: | :--- | :--- |
| `0` | **UNKNOWN** | Unmodeled regions, noise padding, uncertain bits |
| `1` | **SYNC** | Preamble, Barker codes, sync words (e.g. `0xEB90`, `0x1ACFFC1D`) |
| `2` | **HEADER** | Protocol control fields, sequence counters, type IDs |
| `3` | **PAYLOAD** | Information data bytes |
| `4` | **CRC** | Parity, CRC-8/16/32, checksum trailer |
| `5` | **PADDING** | Idle fill bits, zero padding |

---

## 3. Directory Structure

```
astra_bitstream_transformer/
│
├── configs/
│   └── model_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── cnn_encoder.py
│   ├── transformer.py
│   ├── positional_encoding.py
│   ├── heads.py
│   ├── dataset.py
│   ├── augmentations.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── windowing.py
│   ├── merge.py
│   ├── postprocess.py
│   └── utils.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model_shapes.py
│   ├── test_masking.py
│   ├── test_window_merge.py
│   ├── test_loss.py
│   ├── test_regions.py
│   └── test_inference.py
│
├── examples/
│   └── analyze_structure.py
│
├── run_tests.py
├── requirements.txt
└── README.md
```

---

## 4. Usage Example

```python
from astra_bitstream_intelligence.src.inference import BitstreamIntelligenceEngine
from astra_bitstream_transformer.src.inference import BitstreamStructureModel

# 1. Run Stage 12 statistical analysis
stage12_engine = BitstreamIntelligenceEngine()
s12_res = stage12_engine.analyze(recovered_bits)

# 2. Run Stage 13 deep structure prediction
stage13_model = BitstreamStructureModel()
prediction = stage13_model.predict(
    recovered_bits,
    stage12_result=s12_res
)

print(f"Overall Confidence: {prediction.model_confidence:.4f}")
for r in prediction.regions:
    print(f"Bits [{r.start_bit}..{r.end_bit}] -> {r.label} ({r.mean_probability:.2f})")
```

---

## 5. Running Tests

```bash
.venv/Scripts/python -m pytest astra_bitstream_transformer/tests/ -v
```
