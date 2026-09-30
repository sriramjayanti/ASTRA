# ASTRA Stage 9 — FEC Candidate Testing Engine (`astra_fec`)

The **Forward Error Correction (FEC) Candidate Testing Engine** is Stage 9 of the **ASTRA** (Automated Signal Analysis & Recovery Assistant) pipeline. It receives deinterleaved hard bits and soft Log-Likelihood Ratios (LLRs) from Stage 8 (`astra_interleaver`) and tests multiple plausible FEC hypotheses across 5 core code families without assuming a single code type, preserving complete candidate lineage, standardizing decoder metrics, and generating Top-$K$ decoded candidate bitstreams for the downstream **Stage 10 Validation Engine**.

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
       ↓
Interleaver Candidate Testing Engine (Stage 8)
       ↓ [Deinterleaved Hard Bits + Soft LLRs]
STAGE 9: FEC CANDIDATE TESTING ENGINE
       ↓ [Top-K Decoded Candidate Streams + Lineage IDs]
Stage 10: Validation Engine (CRC, Parity, Syndrome, Frame Repetition)
       ↓
Pipeline XGBoost Scorer
       ↓
Bitstream Intelligence & Frame Recovery
```

---

## 2. Core Supported FEC Families

| Family | Decoder Backends | Parameters & Profiles |
| :--- | :--- | :--- |
| **`none`** (NO_FEC) | Uncoded pass-through | Preserves raw stream. Identity decoder metrics. |
| **`convolutional`** | Hard Viterbi & Soft Viterbi | Constraint length $K \in [3, 5, 7]$, rates $1/2, 2/3, 3/4$, generator polynomials in octal, puncturing/depuncturing with neutral LLRs, terminated/continuous traceback. |
| **`reed_solomon`** | Berlekamp-Massey + Chien + Forney over $GF(2^m)$ | $m=8, 6$, standard $RS(255, 223)$, $RS(255, 239)$, $RS(64, 48)$, shortened RS, syndrome clearing, corrected symbol counts. |
| **`ldpc`** | Normalized Min-Sum iterative decoder | Quasi-cyclic parity-check matrices $H$, early stopping on $H \cdot c \equiv 0 \pmod 2$, soft LLR input, syndrome weight telemetry. |
| **`concatenated`** | Reverse-staged decoder (Inner Viterbi $\to$ Outer RS) | Staged multi-layer execution, preserves separate inner/outer telemetry. |

---

## 3. Key Design Rules & Invariants

1. **Rule 1 (Always Include NO_FEC):** Baseline uncoded path is always generated and tested.
2. **Rule 2 (Bounded Profile Registry):** Code parameters are strictly centralized in `configs/fec_profiles.yaml`; no unconstrained blind brute-forcing.
3. **Rule 3 (Soft LLR Priority):** Soft-decision decoding is used wherever soft LLRs are available.
4. **Rule 4 (LLR Polarity Invariant):** Positive LLR indicates bit 0 is favored; negative LLR indicates bit 1 is favored.
5. **Rule 5 (Metric Transparency):** Raw decoder outputs (path metrics, syndrome weights, iteration counts) are standardized into normalized features for downstream pipeline scoring.
6. **Rule 6 (Candidate Lineage):** Every surviving candidate constructs a unique pipeline path ID: `path_{candidate_id}_{variant_id}_{interleaver_id}_{fec_candidate_id}`.
7. **Rule 7 (No Premature Confirmation):** Status values are restricted to `FEC_PLAUSIBLE`, `FEC_WEAK`, `FEC_FAILED`, `FEC_INVALID`. Final confirmation is deferred to Stage 10 CRC and frame validation.

---

## 4. Directory Structure

```
astra_fec/
├── configs/
│   ├── fec_config.yaml                 # Engine settings, weights, beam width
│   └── fec_profiles.yaml               # Centralized code profile registry
├── src/
│   ├── __init__.py                     # Package exports
│   ├── models.py                       # FECProfile, DecoderResult, FECCandidateResult, FECTestResult
│   ├── router.py                       # Family routing and execution
│   ├── profiles.py                     # Profile registry loader and manager
│   ├── no_fec.py                       # NO_FEC uncoded pass-through handler
│   ├── convolutional.py                # Trellis encoder and puncturing/depuncturing
│   ├── viterbi.py                      # Hard & Soft Viterbi decoders
│   ├── reed_solomon.py                 # Galois field engine & Berlekamp-Massey decoder
│   ├── ldpc.py                         # Quasi-cyclic matrix builder & Normalized Min-Sum decoder
│   ├── concatenated.py                 # Reverse-staged concatenated decoder (Viterbi -> RS)
│   ├── candidate_generator.py          # Bounded candidate hypothesis generator with bit offset search
│   ├── metrics.py                      # Multi-family metric normalization
│   ├── scoring.py                      # Rule-based FEC quality scoring
│   ├── pruning.py                      # Beam pruning with guaranteed family diversity
│   ├── validators.py                   # Config & input validation
│   ├── inference.py                    # FECTestingEngine main API
│   └── utils.py                        # Synthetic FEC generator & recall evaluation
├── tests/
│   ├── test_no_fec.py
│   ├── test_viterbi.py
│   ├── test_convolutional_profiles.py
│   ├── test_reed_solomon.py
│   ├── test_concatenated.py
│   ├── test_ldpc.py
│   ├── test_soft_input.py
│   ├── test_candidate_generation.py
│   ├── test_end_to_end_fec.py
│   └── test_fec.py                     # Master 30-test unit suite
├── examples/
│   └── test_fec_candidates.py          # End-to-end demonstration script
├── run_tests.py                        # Test runner
├── requirements.txt
└── README.md
```

---

## 5. Usage Example

```python
from astra_fec.src.inference import FECTestingEngine
import numpy as np

# 1. Initialize Stage 9 Engine
engine = FECTestingEngine()

# 2. Input Deinterleaved Candidate (from Stage 8)
stage8_candidate = {
    "candidate_id": "cand_qpsk_9600",
    "demod_variant_id": "rot90",
    "interleaver_candidate_id": "int_block_r16_c32_0013",
    "hard_bits": np.random.randint(0, 2, size=2048, dtype=np.uint8),
    "soft_llrs": np.random.randn(2048).astype(np.float32)
}

# 3. Test FEC Hypotheses
fec_result = engine.test_candidates(stage8_candidate)

# 4. Access Surviving Candidates for Stage 10 Validation
print(f"Tested {fec_result.tested_candidates_count} hypotheses, {len(fec_result.surviving_candidates)} survived beam.")
for cand in fec_result.surviving_candidates:
    print(f"[{cand.fec_family}] Path: {cand.path_id} | Score: {cand.fec_quality_score:.4f} | Status: {cand.decoder_status.value}")
    # Forward to Stage 10 Validation:
    # cand.decoded_hard_bits
```

---

## 6. Running Tests

```bash
python astra_fec/run_tests.py
```
