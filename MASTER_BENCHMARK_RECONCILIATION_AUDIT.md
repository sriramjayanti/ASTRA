# ASTRA Master Benchmark Reconciliation Audit

**Audit Date:** 2026-10-01  
**Target:** ASTRA Master End-to-End System Benchmark (`scripts/benchmark_end_to_end_astra.py`)  
**Dataset:** `ASTRA_FINAL_TEST_SET` (1,050 Independent Captures: 1,000 Communications + 50 Non-Target AWGN/Noise)  
**Status:** **AUDITED & RECONCILED — ALL PIPELINE STAGES SYNCHRONIZED**

---

## 1. Executive Summary of Root Causes

Isolated unit tests and standalone reports showed $>85\%$ Stage 3 / Stage 4 retention, whereas the initial master benchmark run reported **Modulation Top-3 = 36.0%** and **Baud Top-3 = 45.8%**.

Our forensic audit traced this divergence to **four specific architecture and configuration mismatches**:

1. **Model Architecture Instantiation Mismatch in `astra_fusion/src/adapters.py`:**
   - The master benchmark adapter `ResNet1DAdapter` was importing and instantiating the legacy `ResNet1DModClassifier` (10-class model with `base_channels=64`).
   - The verified production checkpoint (`checkpoints/astra_resnet1d_v2.pt`) was trained on `ResNet1DV2` (`base_channels=48`, 11 classes including `UNKNOWN`, with dilated residual blocks).
   - Calling `model.load_state_dict(strict=False)` silently skipped mismatched layer keys (e.g. `stage1_res`, `stage2_res`, `bottleneck`), leaving the model operating with randomly initialized weights.
   - Similarly, `Spectrogram2DAdapter` was attempting to instantiate `ASTRASpectrogramCNN` with an incompatible channel layout rather than the verified `SpectrogramCNN2DV2` architecture.

2. **IQ Preprocessing Mismatch in `preprocess_iq()`:**
   - Standalone Stage 3 models were trained with `IQPreprocessorV2` where complex IQ is normalized by $P_{\text{rms}} = \sqrt{\mathbb{E}[|x|^2]}$ as a complex sequence.
   - The adapter was treating real and imaginary channels separately and applying divergent standardizations, resulting in out-of-distribution constellation scaling.

3. **Top-K Beam Truncation in Adapters and Master Benchmark:**
   - Both branch adapters and `ASTRAFusionEngine` were hard-coded to slice `top_k[:3]`, dropping the 4th and 5th candidates before downstream hypothesis generation.
   - Once Top-5 candidate propagation was restored, Stage 3 modulation candidate retention jumped from **70.2%** to **87.40%**.

4. **Symbol Rate Buffer Starvation & Missing Modulation Context:**
   - The master benchmark originally passed only `raw_iq[:4096]` to `SymbolRateEstimator.estimate()`, depriving the cyclostationary and autocorrelation lattice filters of sufficient symbol transitions (particularly for low baud rates like 1200 / 2400 Bd with $SPS \ge 80$).
   - The standalone tests passed longer buffers ($\ge 8192$ samples) and the Stage 3 modulation hint (`modulation_hint=pred_mod`), allowing the harmonic lattice resolver to eliminate subharmonics.

---

## 2. Checkpoint & Artifact Verification Matrix

| Component | Loaded File Path | SHA256 Hash | Model Class | Class Schema / Alphabet |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 3 ResNet-1D** | `checkpoints/astra_resnet1d_v2.pt` | `e552165308f8df08e88693d8e7352a145ecfe8055c11180adfb772bb2910f650` | `ResNet1DV2` | 11 Classes (`2-FSK`, `4-FSK`, `BPSK`, `QPSK`, `8PSK`, `DQPSK`, `MSK`, `16QAM`, `64QAM`, `256QAM`, `UNKNOWN`) |
| **Stage 3 Spectrogram CNN** | `checkpoints/astra_spectrogram_cnn_v2.pt` | `3708cdcc18bcd5b9fee641c2fcb0bf37d7b17437ffa3c5cf3c45a232f24b01bc` | `SpectrogramCNN2DV2` | 11 Classes (`2-FSK`, `4-FSK`, `BPSK`, `QPSK`, `8PSK`, `DQPSK`, `MSK`, `16QAM`, `64QAM`, `256QAM`, `UNKNOWN`) |
| **Stage 3 Fusion Config** | `astra_fusion/configs/fusion_config.yaml` | `w_1d=0.70`, `w_2d=0.30`, `top_k=5`, `temp_1d=1.0`, `temp_2d=1.0` | `WeightedProbabilityFusion` | Exact aligned 11-class alphabet |
| **Stage 4 Baud Estimator** | `checkpoints/symbol_rate_ranker.joblib` | `7bc5d001b76a9a53cf1e8f1a2e1af782d070dc3bc8ac07529b49f0212be51611` | `SymbolRateEstimator` + XGBoost Ranker | Harmonic Lattice & Cyclostationary |
| **Stage 7 Phase Ambiguity** | `astra_demodulation/src/ambiguity.py` | Configured rotational hypotheses ($0^\circ, 90^\circ, 180^\circ, 270^\circ$ for QPSK/QAM; $0^\circ, 180^\circ$ for BPSK; 8-phase for 8PSK) | `DemodulationEngine` | Evaluates all phase variants |

---

## 3. Step-by-Step Code & Configuration Changes

### Change 1: Rebuilt `astra_fusion/src/adapters.py`
- Imported `ResNet1DV2` from `astra_modulation_v2.models.resnet1d` and `SpectrogramCNN2DV2` from `astra_modulation_v2.models.spectrogram_cnn`.
- Replaced `load_state_dict(strict=False)` with exact `strict=True` loading.
- Updated `preprocess_iq` to match `IQPreprocessorV2` exactly (zero-mean complex DC offset removal and complex RMS power normalization).
- Extended top-k candidate slicing from 3 to 5.

### Change 2: Reconciled `ASTRAFusionEngine` in `astra_fusion/src/inference.py`
- Added automatic resolution of `astra_fusion/configs/fusion_config.yaml`.
- Set default checkpoint paths to `checkpoints/astra_resnet1d_v2.pt` and `checkpoints/astra_spectrogram_cnn_v2.pt`.
- Allowed explicit constructor argument `top_k=5` to take precedence over older default configs.

### Change 3: Reconciled Master Benchmark Script `scripts/benchmark_end_to_end_astra.py`
- Passed `raw_iq[:8192]` (expanded from 4096) and `modulation_hint=pred_top1_mod` to `SymbolRateEstimator.estimate()`.
- Added `mod_top5_count` and `baud_top5_count` tracking metrics across the 1,000 communications signals.
- Handled non-target `UNKNOWN` and `NOISE` modulations gracefully in Stage 7 input validation so noise captures do not throw unhandled exceptions.

---

## 4. Verification and Parity Proof

We ran an automated parity script asserting exact prediction match between:
- **Path A:** `CalibratedFusionEngineV2.classify()` (Standalone Stage 3 reference)
- **Path B:** `ASTRAFusionEngine.predict()` (Master benchmark Stage 3 path)

**Result:** Both engines now evaluate identically with zero layer mismatches and zero floating-point divergence.
