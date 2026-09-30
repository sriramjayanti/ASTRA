# ASTRA Constellation Analysis Engine

**Automated Signal Analysis & Recovery Assistant (ASTRA)**  
*Unsupervised Spatial Clustering, Radial Ring Analysis, and Geometric EVM Support*

---

## 1. Overview & Role in ASTRA

The **Constellation Analysis Engine** performs spatial and geometric clustering on normalized I/Q complex scatter to extract supporting modulation evidence:

```
Complex IQ Scatter
         ↓
Normalization & Coordinate Transformation ([N, 2] Real Plane)
         ↓
├── Multi-K K-Means (Silhouette & Inertia for K in [2, 4, 8, 16, 64])
├── DBSCAN Density Clustering (Natural cluster count & noise ratio)
├── Radial Ring Analysis (Amplitude histogram energy levels)
├── Angular Phase Uniformity (Modulo-2pi/M phase dispersion)
└── Template EVM Calculation (RMS Error Vector Magnitude against ideal templates)
         ↓
ConstellationPrediction (Best Type, Order M, Confidence, EVM, Status)
         ↓
Candidate / Hypothesis Engine (Stage 5)
```

---

## 2. Key Metrics & Evidence

| Metric | Mechanism | Interpretation |
| :--- | :--- | :--- |
| **K-Means Silhouette** | Multi-K evaluation ($K=2, 4, 8, 16, 64$) | Peak Silhouette score indicates true constellation order $M$. |
| **DBSCAN Clusters** | Density-based unconstrained clustering | Discovers natural dense regions without assuming $K$ a priori. |
| **Radial Ring Count** | Peak detection on amplitude histogram | 1 ring for PSK; 3 rings for 16-QAM; $\ge 6$ rings for 64-QAM. |
| **Constant Modulus Variance**| $\text{Var}(\|x[n]\| / \mu)$ | $\approx 0$ for PSK/FSK; $>0.05$ for multi-amplitude QAM. |
| **Angular Uniformity** | Phase dispersion over $2\pi/M$ sectors | High ($\ge 0.8$) for clean PSK constellations. |
| **Square Grid Compactness** | Kurtosis deviation of I and Q lattices | High ($\ge 0.7$) for clean Cartesian QAM constellations. |
| **Template EVM (%)** | Nearest-neighbor distance to ideal points | Error Vector Magnitude percentage. |

---

## 3. Quickstart API Usage

```python
import numpy as np
from astra_constellation.src.inference import ConstellationAnalyzer

analyzer = ConstellationAnalyzer()

# Analyze IQ samples
iq_samples = np.fromfile("capture.iq", dtype=np.complex64)
pred = analyzer.analyze(iq_samples)

print(f"Best Constellation : {pred.best_constellation} (Order M = {pred.m_ary_order})")
print(f"Confidence         : {pred.confidence:.2f}")
print(f"Status             : {pred.status}")
print(f"Estimated EVM      : {pred.evidence['estimated_evm_percent']:.2f}%")
```
