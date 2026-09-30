# ASTRA Symbol-Rate (Baud) Estimation Engine

**Automated Signal Analysis & Recovery Assistant (ASTRA)**  
*Production-Ready Multi-Modal DSP + XGBoost Hybrid Symbol-Rate Estimation Engine*

---

## 1. Overview & Core Principles

The **ASTRA Symbol-Rate Estimation Engine** accurately estimates the **symbol rate** (also known as **baud rate**, $R_s$) and the float **samples-per-symbol** ($\text{SPS} = F_s / R_s$) of incoming raw IQ bursts across arbitrary modulation schemes and channel impairments.

```
Raw IQ Burst / Window
         ↓
Normalization & Preprocessing
         ↓
Multi-Modal DSP Evidence Extraction (10+ independent estimators)
         ↓
Candidate Hypothesis Generator (Clustering ±2%, Harmonics, Standard Priors)
         ↓
Candidate-Level Feature Matrix Builder (20 Engineered Features)
         ↓
XGBoost Candidate Ranker (Leakage-Free Group Validation)
         ↓
Confidence & ASTRA Status Assignment (CONFIRMED / ESTIMATED / POSSIBLE / UNKNOWN)
         ↓
Top-K Baud Hypotheses + Float SPS + Preserved DSP Evidence
         ↓
Candidate / Hypothesis Engine (Stage 4)
```

### Why Hybrid DSP + XGBoost? (Never a Black-Box Baud Predictor)
1. **Explainability**: Pure neural networks predicting raw baud rates from raw IQ fail on non-standard rates, hallucinate under frequency offsets (CFO), and offer zero explainability.
2. **Robustness & Uncertainty Preservation**: Multiple independent DSP algorithms extract physical periodicities and spectral properties. The candidate generator proposes candidate rates ($R_s$). XGBoost is tasked *only* with ranking the likelihood of correctness for each hypothesis based on multi-source evidence.
3. **Graceful Degradation**: If no signal periodicity exists (pure noise or severe interference), the engine produces an explicit `UNKNOWN` status rather than forcing a false rate.

---

## 2. Symbol Rate (Baud) vs. Bit Rate

| Parameter | Symbol Rate ($R_s$, Baud) | Bit Rate ($R_b$, bps) | Formula |
| :--- | :--- | :--- | :--- |
| **BPSK** | 9600 Baud | 9600 bps | $R_b = R_s \times 1$ |
| **QPSK** | 9600 Baud | 19200 bps | $R_b = R_s \times 2$ |
| **8PSK** | 9600 Baud | 28800 bps | $R_b = R_s \times 3$ |
| **16-QAM** | 9600 Baud | 38400 bps | $R_b = R_s \times 4$ |
| **64-QAM** | 9600 Baud | 57600 bps | $R_b = R_s \times 6$ |

> **Crucial Rule**: This engine measures **Symbol Rate (Baud)** and **Samples-Per-Symbol (SPS)**. Uncoded and coded bit rates are resolved downstream in the Demodulation and FEC engines.

---

## 3. Required Multi-Modal DSP Estimators

The engine implements 10 independent DSP evidence streams:

1. **Envelope Autocorrelation**: Unbiased autocorrelation on $a[n] = |x[n]|$, identifying periodic amplitude dips at symbol boundaries.
2. **Magnitude-Squared Power Autocorrelation**: Autocorrelation on $p[n] = |x[n]|^2$.
3. **Instantaneous Frequency Autocorrelation**: Computes phase difference $\Delta\phi[n] = \arg(x[n] \cdot x^*[n-1])$ and tracks frequency transition periodicity (specialized for 2-FSK / 4-FSK / MSK).
4. **Spectral Peak Spacing**: Welch PSD analysis of non-linear power transformations ($x^2[n]$) to expose clock lines at $\pm R_s/2$.
5. **PSD & Welch Spectral Features**: Wiener spectral flatness, spectral centroid, and noise-floor estimates.
6. **Cyclostationary Cyclic Autocorrelation**: Evaluates cyclic frequency peaks $\alpha = R_s$ via FFT of the zero-mean squared envelope.
7. **Bandwidth-Derived RRC Hypotheses**: Measures 99% occupied bandwidth ($\text{OBW}$) and generates candidate rates across root-raised-cosine rolloff hypotheses $\alpha \in \{0.20, 0.25, 0.35, 0.50\}$:
   $$R_s = \frac{\text{OBW}}{1 + \alpha}$$
8. **Harmonic Relationship Detection**: Flags and relates $2\times, 0.5\times, 4\times, 0.25\times$ harmonic multiples to resolve clock aliasing.
9. **Multi-Source Support Scoring**: Counts how many independent DSP streams validate each rate.
10. **Standard Rate Grid Priors**: Soft priors for standard rates (e.g. 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200) without restricting arbitrary continuous rates (e.g. 7350, 14200 Baud).

---

## 4. Candidate Feature Schema (`v1.0.0`)

Every candidate hypothesis $R_s$ is converted into a 20-dimensional feature vector:

| # | Feature Name | Description |
| :--- | :--- | :--- |
| 1 | `candidate_rate_hz` | Proposed symbol rate $R_s$ (Hz) |
| 2 | `candidate_normalized` | Normalized rate $R_s / F_s$ |
| 3 | `samples_per_symbol` | Float oversampling ratio $F_s / R_s$ |
| 4 | `autocorr_score` | Peak prominence from envelope autocorrelation |
| 5 | `power_autocorr_score`| Peak prominence from power autocorrelation |
| 6 | `cyclostationary_score`| Normalized cyclic autocorrelation peak strength |
| 7 | `bandwidth_consistency`| Alignment between occupied bandwidth and $R_s$ |
| 8 | `spectral_peak_score` | Spacing regularity in squared spectrum |
| 9 | `instant_freq_score` | Peak prominence in instantaneous frequency track |
| 10 | `harmonic_consistency`| Indicator of detected harmonic relationship |
| 11 | `support_count` | Number of independent estimators supporting $R_s$ |
| 12 | `is_standard_grid` | Proximity ($\le 3\%$) to standard telecom baud rates |
| 13 | `snr_db_estimate` | Fourth-moment ($M_2M_4$) SNR estimate (dB) |
| 14 | `clipping_ratio` | Percentage of saturated ADC samples |
| 15 | `spectral_flatness` | Wiener entropy of PSD |
| 16 | `cfo_magnitude_hz` | Estimated carrier frequency offset (Hz) |
| 17 | `mod_fsk_family_prob` | Aggregated probability of FSK family (optional) |
| 18 | `mod_psk_family_prob` | Aggregated probability of PSK family (optional) |
| 19 | `mod_qam_family_prob` | Aggregated probability of QAM family (optional) |
| 20 | `mod_top1_confidence` | Fusion Engine top-1 confidence (optional) |

---

## 5. Group-Level Splitting & Anti-Leakage Guarantee

All candidate rows generated from a single IQ burst are strictly tied to that signal's `signal_id`. Train / Validation / Test partitions are performed using `GroupShuffleSplit` on `signal_id`. This prevents candidate-row leakage and guarantees honest out-of-sample generalization.

---

## 6. Python API & Quickstart

```python
import numpy as np
from astra_symbol_rate.src.inference import SymbolRateEstimator

# 1. Initialize Estimator (auto-loads trained checkpoint)
estimator = SymbolRateEstimator()

# 2. Estimate from Raw IQ (e.g. 192 kHz sample rate)
iq_samples = np.fromfile("capture.iq", dtype=np.complex64)
prediction = estimator.estimate(
    iq=iq_samples,
    sample_rate_hz=192000.0,
    modulation_prediction={"top_k": [{"class": "QPSK", "probability": 0.85}]},  # Optional
)

# 3. Access Top-K Results and SPS
print(f"Status: {prediction.status}")
print(f"Best Rate: {prediction.best_symbol_rate_hz:.1f} Baud (SPS = {prediction.samples_per_symbol:.2f})")
print(f"Confidence: {prediction.confidence:.2f}")

for cand in prediction.top_k:
    print(f"  Rank #{cand['rank']}: {cand['rate_hz']} Baud | Score={cand['score']} | Supported by: {cand['supported_by']}")
```

---

## 7. ASTRA Status & Downstream Handoff

The prediction output contains explicit status tiers:
- **`CONFIRMED`**: Top score $\ge 0.80$, margin $\ge 0.18$, supported by $\ge 2$ estimators.
- **`ESTIMATED`**: High confidence rate candidate ($\ge 0.50$).
- **`POSSIBLE`**: Lower confidence candidate ($\ge 0.28$).
- **`UNKNOWN`**: Low score, high candidate disagreement, or pure noise.

This feeds directly into **ASTRA Stage 4: Candidate / Hypothesis Engine**, forming joint pairs:
$$\{\text{Modulation Candidate}\} \times \{\text{Symbol Rate Candidate}\} \to \text{Synchronization \& Demodulation Parameters}$$
