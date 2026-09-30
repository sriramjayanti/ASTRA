# ASTRA Stage 5 — Candidate / Hypothesis Engine

## Overview
The **Candidate / Hypothesis Engine** is Stage 5 of the **ASTRA (Automated Signal Analysis & Recovery Assistant)** signal processing pipeline.

ASTRA operates under a core foundational principle: **Never trust only Top-1 predictions**. Signal parameters under low SNR, severe multipath, and Doppler shifts carry inherent uncertainty. Instead of committing prematurely to a single modulation scheme or baud rate, Stage 5 combines uncertain evidence from upstream models and constructs a bounded, ranked, and beam-pruned Cartesian grid of **Receiver Hypotheses**:

$$\text{Receiver Hypotheses} = \text{Top-}K_{\text{Modulation}} \times \text{Top-}K_{\text{Symbol Rate}}$$

Each hypothesis carries full source evidence, modulation-aware synchronization routing hints, auditable lifecycle tracking, and prepares downstream hooks for **Synchronization (Stage 6)** and **Demodulation (Stage 7)**.

---

## Pipeline Architecture

```
1. Modulation Fusion (1D + 2D)        2. Symbol-Rate Estimation (DSP + XGBoost)
        (Top-K Modulations)                   (Top-K Symbol Rates)
                 \                                     /
                  \                                   /
   4. Random Forest Support (Optional)      3. Constellation Support (Optional)
                   \                                 /
                    \                               /
                     ▼                             ▼
             ┌──────────────────────────────────────────────┐
             │       STAGE 5: CANDIDATE / HYPOTHESIS        │
             │                   ENGINE                     │
             │                                              │
             │   1. Parse & validate inputs                 │
             │   2. Deduplicate / merge close baud rates    │
             │   3. Generate Cartesian grid (Mod × Baud)    │
             │   4. Multi-evidence prior scoring            │
             │   5. Physical validity checks (SPS, Nyquist) │
             │   6. Soft beam pruning (Top beam_width)      │
             │   7. Attach sync / demod routing hints       │
             │   8. Immutable source evidence & history     │
             └──────────────────────┬───────────────────────┘
                                    │
                                    ▼
                          CandidateSet (Ranked)
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
         Stage 6: Synchronization        Stage 7: Demodulation
```

---

## Core Capabilities

1. **Cartesian Grid Generation:** Generates Top-$K_{\text{mod}} \times \text{Top-}K_{\text{rate}}$ combinations (e.g., $3 \times 3 = 9$).
2. **Multi-Evidence Prior Scoring:**
   $$\text{Score} = (P_{\text{mod}})^{w_{\text{mod}}} \times (P_{\text{rate}})^{w_{\text{rate}}} \times (\text{RF}_{\text{fam}})^{w_{\text{rf}}} \times (\text{Constellation})^{w_{\text{const}}}$$
   - Pre-sync vs. post-sync constellation stage weighting.
   - Optional auxiliary evidence never penalizes if missing.
3. **Physical Feasibility & Hard Rejection:**
   - Enforces $F_s > R_s$ (Nyquist condition: $\text{SPS} \ge 1.0$).
   - Calculates exact float Samples-Per-Symbol ($\text{SPS} = F_s / R_s$).
   - Rejects non-finite/negative rates with auditable rejection reasons.
4. **Rate Deduplication:** Merges near-duplicate baud estimates within configured tolerance (default $\pm 2\%$).
5. **Beam Search Pruning:** Retains top `beam_width` hypotheses active while keeping lower ranks recorded.
6. **Fallback Support:** Fallback to standard standard baud grids ($1200, 2400, 4800, 9600, 19200, \dots$) when baud estimation is unavailable.
7. **Modulation-Aware Synchronization Routing Hints:**
   - **FSK:** Discriminator / Envelope detection, Gaussian/Rectangular matched filtering, Early-Late / Zero-Crossing timing.
   - **PSK:** Costas loop (orders 2, 4, 8), RRC matched filtering, Gardner / Mueller-Müller timing.
   - **QAM:** Decision-directed PLL, RRC matched filtering, Gardner / Mueller-Müller timing.
8. **Memory Efficiency:** Candidates reference signal identifiers (`signal_id`), **never copying large IQ buffers**.

---

## Directory Structure

```
astra_candidate_engine/
├── configs/
│   └── candidate_config.yaml         # Configuration parameters
├── src/
│   ├── __init__.py                   # Package exports
│   ├── models.py                     # ReceiverHypothesis & CandidateSet dataclasses
│   ├── mappings.py                   # Family, demodulator, and sync routing hints
│   ├── grid_generator.py             # Cartesian product & rate variants generator
│   ├── scoring.py                    # Multi-evidence prior scoring functions
│   ├── pruning.py                    # Beam pruning, duplicate rate merge, invalid checks
│   ├── validators.py                 # Input and physical validation rules
│   ├── lifecycle.py                  # Lifecycle state tracking and expansion hooks
│   ├── explainability.py             # Explainability breakdown & summary table
│   ├── inference.py                  # CandidateHypothesisEngine main API
│   └── utils.py                      # Synthetic mock generators for testing
├── tests/
│   ├── test_candidate_engine.py      # Master 20-in-1 test suite
│   ├── test_grid.py                  # Grid generation unit tests
│   ├── test_scoring.py               # Scoring unit tests
│   ├── test_pruning.py               # Pruning unit tests
│   ├── test_duplicates.py            # Duplicate rate merge unit tests
│   ├── test_unknown.py               # Unknown & fallback mode unit tests
│   └── test_candidate_schema.py      # Schema & lifecycle unit tests
├── examples/
│   └── generate_candidates.py        # End-to-end 3x3 grid demonstration script
├── run_tests.py                      # Test runner
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## Quickstart

### 1. Basic Generation

```python
from astra_candidate_engine.src.inference import CandidateHypothesisEngine

engine = CandidateHypothesisEngine()

fusion_prediction = {
    "top_k": [
        {"class": "QPSK", "probability": 0.80},
        {"class": "8PSK", "probability": 0.12},
        {"class": "16-QAM", "probability": 0.05}
    ]
}

symbol_rate_prediction = {
    "top_k": [
        {"symbol_rate_hz": 9600.0, "score": 0.85},
        {"symbol_rate_hz": 4800.0, "score": 0.10},
        {"symbol_rate_hz": 19200.0, "score": 0.05}
    ]
}

candidate_set = engine.generate(
    modulation_prediction=fusion_prediction,
    symbol_rate_prediction=symbol_rate_prediction,
    sample_rate_hz=192000.0,
    signal_id="SIG_001"
)

# Print GUI / Terminal summary table
print(engine.print_summary(candidate_set))
```

### 2. Running Demonstration

```bash
python astra_candidate_engine/examples/generate_candidates.py
```

### 3. Running Unit Tests

```bash
python astra_candidate_engine/run_tests.py
```
All 20 unit tests pass with 100% test coverage.
