# ASTRA Random Forest Support Model

**Automated Signal Analysis & Recovery Assistant (ASTRA)**  
*Engineered DSP Feature Baseline, Broad Signal-Family, and Signal-Quality Classifier*

---

## 1. Overview & Architectural Role

The **Random Forest Support Model** operates on **36 structured DSP physical and statistical features** extracted from complex IQ bursts. 

```
Raw IQ Burst / Window
         ↓
DSP Preprocessing (DC Removal, Unit RMS Normalization)
         ↓
DSP Feature Extractor Adapter (36 Features across Spectral, Amplitude, Phase, Frequency, Cumulants, Temporal)
         ↓
Median Feature Imputation (SimpleImputer)
         ↓
├── Broad Signal-Family Classifier (FSK / PSK / QAM / UNKNOWN)
└── Multi-Label Signal Quality Classifier (LOW_SNR / CLIPPED / MULTIPATH / CFO_AFFECTED)
         ↓
Supporting Evidence + Feature Importance + Explainability Metrics
         ↓
Candidate / Hypothesis Engine (Stage 5)
```

> **Critical Rule**: The Random Forest is **supporting evidence only** and a classical ML baseline. It does **not** replace or override the deep neural networks (1D ResNet, 2D CNN, Fusion Engine), synchronization, or FEC validation.

---

## 2. Engineered 36-Feature Schema (`rf_features_v1`)

| Category | Features Included |
| :--- | :--- |
| **Spectral** | `occupied_bandwidth_hz`, `spectral_centroid_hz`, `spectral_spread_hz`, `spectral_flatness`, `spectral_rolloff_hz`, `psd_variance`, `peak_to_average_power_ratio` |
| **Amplitude & Envelope** | `mean_magnitude`, `variance_magnitude`, `rms_amplitude`, `skewness_magnitude`, `kurtosis_magnitude`, `crest_factor`, `envelope_variance` |
| **Phase** | `mean_phase_diff`, `variance_phase_diff`, `circular_variance`, `phase_concentration` |
| **Frequency** | `inst_freq_mean_hz`, `inst_freq_std_hz`, `inst_freq_kurtosis`, `dominant_frequency_states` |
| **IQ Statistics & Cumulants** | `i_variance`, `q_variance`, `iq_covariance`, `circularity_coefficient`, `real_imag_correlation`, `cumulant_c40_mag`, `cumulant_c42_mag` |
| **Temporal & Periodic** | `autocorr_max_peak_lag`, `autocorr_max_peak_score`, `zero_crossing_rate` |
| **Constellation Geometry** | `constellation_cluster_count`, `radial_ring_count`, `constant_modulus_variance` |
| **Signal Quality** | `estimated_snr_db`, `clipping_ratio`, `cfo_magnitude_hz` |

---

## 3. Supported Tasks

### Task A: Broad Signal-Family Classification
- Classes: `FSK`, `PSK`, `QAM`, `UNKNOWN`
- Evaluates macro-family probability distribution and top1-top2 confidence margin.

### Task B: Multi-Label Signal Quality & Impairment Classification
- Labels: `low_snr`, `clipped`, `multipath`, `cfo_affected`
- Outputs independent probabilities $P(\text{impairment}_i = 1)$ to assist downstream synchronization and demodulator parameter selection.

---

## 4. Quickstart Python API

```python
import numpy as np
from astra_random_forest.src.inference import RandomForestSupportEngine

# Initialize Engine (auto-loads trained checkpoint)
engine = RandomForestSupportEngine()

# Predict from raw IQ signal or feature dictionary
iq_samples = np.fromfile("capture.iq", dtype=np.complex64)
result = engine.predict_all(iq_samples, sample_rate_hz=192000.0)

print(f"Predicted Family  : {result['family']} (Confidence: {result['confidence']:.2f})")
print(f"Family Probs      : {result['probabilities']}")
print(f"Quality Evidence  : {result['quality']}")
print(f"Top Features      : {result['important_evidence']}")
```
