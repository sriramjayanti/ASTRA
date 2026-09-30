# ASTRA Stage 11 — Pipeline Scoring Model

Production-ready XGBoost-based Pipeline Scoring Model that receives complete candidate-path telemetry across Stages 5 through 10 and ranks the competing signal-recovery pipelines.

## Overview

By Stage 11, ASTRA has accumulated multi-stage evidence for multiple complete candidate recovery paths:
$$\text{Modulation} \longrightarrow \text{Baud} \longrightarrow \text{Sync} \longrightarrow \text{Demod} \longrightarrow \text{Interleaver} \longrightarrow \text{FEC} \longrightarrow \text{Validation}$$

Stage 11 does not rely on early confidence or single metrics alone. It combines:
1. **Stage 5**: Candidate priors, 1D/2D fusion logits, RF support, constellation geometry
2. **Stage 6**: Sync metrics (CFO residual, Gardner timing lock, Costas phase error)
3. **Stage 7**: Demodulation metrics (EVM %, decision margins, soft LLR quality)
4. **Stage 8**: Interleaver evidence (structural score, permutation alignment)
5. **Stage 9**: FEC telemetry (normalized Viterbi path metric, RS syndrome weight, LDPC convergence)
6. **Stage 10**: Validation metrics (CRC pass rate, sync periodicity, header consistency, length check, re-encoding match)
7. **Cross-Stage Consistency**: Agreement between modulation, constellation, timing lock, demod quality, and CRC.

## Architecture

```
Candidate Trees (Stages 5-10 Telemetry)
                  │
                  ▼
    ┌───────────────────────────┐
    │  PipelineFeatureBuilder   │ ──> Fixed Schema: pipeline_features_v1 (60+ features)
    └─────────────┬─────────────┘
                  │
                  ▼
    ┌───────────────────────────┐
    │  XGBoost Ranking Model    │ ──> Scale-pos-weight & Early Stopping
    └─────────────┬─────────────┘
                  │
                  ▼
    ┌───────────────────────────┐
    │  Probability Calibrator   │ ──> Isotonic Regression / Platt Scaling
    └─────────────┬─────────────┘
                  │
                  ▼
        PipelineRankingResult
   ├── Top-K Ranked Candidate Paths
   ├── Score Margin (Top-1 vs Top-2)
   ├── Ranking Uncertainty (Entropy)
   ├── Confidence Tier (CONFIRMED / ESTIMATED / POSSIBLE / UNKNOWN)
   └── Key Evidence Drivers (Positive / Negative)
```

## Running Tests

```bash
python -m pytest -v astra_pipeline_scorer/tests
```
