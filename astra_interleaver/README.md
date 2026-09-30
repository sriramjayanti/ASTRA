# ASTRA Stage 8 — Interleaver Candidate Testing Engine (`astra_interleaver`)

The **Interleaver Candidate Testing Engine** is Stage 8 of the **ASTRA** (Automated Signal Analysis & Recovery Assistant) pipeline. It receives one or more demodulated bitstream variants from Stage 7 and tests plausible deinterleaving hypotheses across 5 core interleaver families without changing bit values, preserving exact hard bit / soft LLR permutation alignments, scoring structural evidence, beam-pruning weak hypotheses, and generating Top-$K$ candidate deinterleaved streams for downstream **Stage 9 FEC testing**.

---

## 1. Pipeline Context

```
Signal Capture
       ↓
DSP Preprocessing
       ↓
1D ResNet + 2D Spectrogram CNN
       ↓
Multi-Branch Modulation Fusion (Stage 1)
       ↓
Symbol-Rate Estimator (Stage 2)
       ↓
Constellation & EVM Analysis (Stage 3)
       ↓
Random Forest Classifier (Stage 4)
       ↓
Candidate / Hypothesis Engine (Stage 5)
       ↓
Synchronization Engine (Stage 6)
       ↓
Demodulation Engine (Stage 7)
       ↓ [Hard Bits + Soft LLRs + Phase Variants]
STAGE 8: INTERLEAVER CANDIDATE TESTING ENGINE
       ↓ [Top-K Deinterleaved Candidate Streams]
Stage 9: FEC Candidate Testing Engine
       ↓
Validation Engine
       ↓
Pipeline Scorer
```

---

## 2. Core Supported Interleaver Families

| Family | Description | Parameter Space |
| :--- | :--- | :--- |
| **`identity`** | Pass-through (`NO_INTERLEAVER`). Preserves raw stream. | Length $N$ |
| **`block`** | Rectangular matrix interleaver. Writes row-wise, reads column-wise (or inverse). | Rows $R$, Columns $C$, Orientation, Padding policy |
| **`convolutional`** | Ramsey/Forney convolutional delay-line commutator. Stateful across chunks. | Branch count $B$, Delay step $M$, Latency |
| **`helical`** | 2D rectangular buffer diagonal traversal. | Rows $R$, Columns $C$, Step $s$, Orientation |
| **`pseudo_random`** | Constrained pseudo-random permutation. | Algorithm (`pcg64`, `fisher_yates`), Seed, Profile registry |

---

## 3. Key Design Rules & Invariants

1. **Rule 1 (Always Include Identity):** Candidate set always includes `NO_INTERLEAVER`.
2. **Rule 2 (Bit Value Invariance):** Interleaving only modifies bit order, never bit value.
3. **Rule 3 (Hard/Soft Alignment):** For any permutation mapping $p$, `deinterleaved_hard_bits[i]` and `deinterleaved_soft_llrs[i]` are reordered using the exact same inverse gather indices.
4. **Rule 4 (Bounded Search):** Candidate grids are bounded by divisibility heuristics and search limits to avoid combinatorial explosion.
5. **Rule 5 (Constrained Pseudo-Random Search):** Blind brute force over large seed spaces is strictly prohibited; search operates over constrained seed lists and registered profile catalogs.
6. **Rule 6 (Stateful Convolutional Support):** `ConvolutionalDeinterleaverState` tracks shift-register contents across streaming chunks.
7. **Rule 7 (Permutation Validation):** Every finite block mapping is mathematically validated for injectivity, surjectivity, and range bounds $[0, N-1]$.
8. **Rule 8 (No Premature Confirmation):** Status values are limited to `INTERLEAVER_PLAUSIBLE`, `INTERLEAVER_WEAK`, `INTERLEAVER_INVALID`, `INTERLEAVER_REJECTED`. Final confirmation is deferred to Stage 9 FEC and CRC validation.

---

## 4. Directory Structure

```
astra_interleaver/
├── configs/
│   └── interleaver_config.yaml         # Configurable search grids & scoring weights
├── src/
│   ├── __init__.py                     # Package exports
│   ├── models.py                       # Data models, enums, dataclasses
│   ├── router.py                       # Family routing and execution
│   ├── identity.py                     # Pass-through identity deinterleaver
│   ├── block.py                        # Rectangular block deinterleaver
│   ├── convolutional.py                # Stateful Ramsey/Forney convolutional engine
│   ├── helical.py                      # Helical diagonal permutation engine
│   ├── pseudo_random.py                # Constrained pseudo-random permutation engine
│   ├── permutation.py                  # Core permutation math, inversion & LLR alignment
│   ├── candidate_generator.py          # Bounded hypothesis generation & deduplication
│   ├── structural_features.py          # Autocorrelation, periodicity, entropy, run lengths
│   ├── scoring.py                      # Transparent rule-based structural scoring
│   ├── pruning.py                      # Beam pruning & duplicate permutation removal
│   ├── validators.py                   # Configuration and demod input validation
│   ├── inference.py                    # InterleaverTestingEngine main API
│   └── utils.py                        # Synthetic test generator & recall evaluation
├── tests/
│   ├── test_identity.py
│   ├── test_block.py
│   ├── test_convolutional.py
│   ├── test_helical.py
│   ├── test_pseudo_random.py
│   ├── test_llr_alignment.py
│   ├── test_scoring.py
│   ├── test_end_to_end_interleaver.py
│   └── test_interleaver.py             # Master 30-test unit suite
├── examples/
│   └── test_interleaver_candidates.py  # End-to-end demo script
├── run_tests.py                        # Test runner
├── requirements.txt
└── README.md
```

---

## 5. Usage Example

```python
from astra_interleaver.src.inference import InterleaverTestingEngine
import numpy as np

# 1. Initialize Stage 8 Engine
engine = InterleaverTestingEngine()

# 2. Input Demodulation Variant (from Stage 7)
demod_variant = {
    "candidate_id": "cand_qpsk_9600",
    "variant_id": "rot90",
    "hard_bits": np.random.randint(0, 2, size=2048, dtype=np.uint8),
    "soft_llrs": np.random.randn(2048).astype(np.float32)
}

# 3. Test Deinterleaver Hypotheses
result = engine.test_candidates(demod_variant)

# 4. Access Surviving Top-K Hypotheses
print(f"Tested {result.tested_candidates_count} candidates, {len(result.surviving_candidates)} survived beam.")
for cand in result.surviving_candidates:
    print(f"[{cand.interleaver_family}] ID={cand.interleaver_candidate_id} Score={cand.overall_interleaver_score:.4f}")
    # Forward to Stage 9 FEC testing:
    # cand.deinterleaved_hard_bits
    # cand.deinterleaved_soft_llrs
```

---

## 6. Running Tests

```bash
python astra_interleaver/run_tests.py
```
