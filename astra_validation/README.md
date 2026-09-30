# ASTRA Stage 10 — Validation Engine

Production-grade multi-mechanism validation engine that receives decoded candidate bitstreams from Stage 9 Forward Error Correction (FEC) testing and evaluates whether each candidate recovery path is internally consistent.

## Overview

A candidate pipeline (Modulation $\rightarrow$ Baud $\rightarrow$ Sync $\rightarrow$ Demod $\rightarrow$ Interleaver $\rightarrow$ FEC) can decode bits without crashing, but only a genuine signal will exhibit independent structural consistency. Stage 10 gathers independent downstream evidence rather than trusting model confidence or single metrics alone.

### Primary Validation Mechanisms

1. **CRC Verification**:
   - Parameterized Rocksoft CRC engine supporting CRC-8 (ATM, Bluetooth), CRC-16 (CCITT-FALSE, IBM, XMODEM), and CRC-32 (IEEE, BZIP2).
   - Multi-hypothesis search across candidate frame lengths and bit offsets.
   - Evaluates multi-frame pass rate to eliminate random false positives.
2. **Parity Checks**:
   - Even, odd, and 2D block parity consistency evaluations.
3. **FEC Syndrome & Metric Integration**:
   - Normalizes Stage 9 telemetry (RS syndromes, LDPC $H \cdot c \equiv 0 \pmod 2$, Viterbi path metrics) into normalized $[0, 1]$ evidence.
4. **Frame Repetition & Periodicity**:
   - Autocorrelation peak estimation and inter-frame pairwise similarity across segmented candidates.
5. **Sync-Word Detection**:
   - Exact and Hamming distance-tolerant matching (e.g. `0xEB90`, `0x1ACFFC1D`) and regular spacing consistency.
6. **Header Consistency Rules**:
   - Schema-driven validation for constant fields, sequence counters with wrap-around, and bounded integers.
7. **Length-Field Consistency**:
   - Compares declared length fields against physical frame boundaries.
8. **FEC Re-encoding Verification**:
   - Re-encodes message bits with candidate FEC parameters and compares against received deinterleaved channel bits.
9. **Independent Evidence Grouping & Contradiction Tracking**:
   - Groups evidence into `FEC_INTERNAL`, `CRC`, `FRAME_PERIODICITY`, `SYNC`, `HEADER_STRUCTURE`, `LENGTH_CONSISTENCY`, `REENCODING`, and `GENERIC_STRUCTURE`.
   - Records hard contradictions and applies penalties.
10. **Fixed Stage 11 Feature Schema (`validation_features_v1`)**:
    - Exports structured, fixed-order feature vectors for Stage 11 XGBoost Pipeline Scorer training and inference.

## Architecture

```
Stage 9 Decoded Candidate Bitstreams (FECCandidateResult)
                         │
                         ▼
             ┌───────────────────────┐
             │   ValidationEngine    │
             └───────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    [CRC Engine]    [Sync & Headers] [FEC Telemetry]
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
        [Evidence Grouping & Contradiction Tracking]
                         │
                         ▼
                 ValidationResult
          ├── validation_status (STRONG / MODERATE / WEAK / FAILED / INCONCLUSIVE)
          ├── overall_validation_score [0.0 - 1.0]
          └── validation_features_v1 (for Stage 11 XGBoost)
```

## Running Tests

```bash
python -m pytest -v astra_validation/tests
```
