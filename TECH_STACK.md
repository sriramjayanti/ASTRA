# ASTRA Technology Stack & Module Mapping

This document provides a comprehensive audit of the core software libraries, scientific frameworks, and computational technologies employed across the ASTRA platform, along with their design rationale and component locations.

---

## Technology Summary Matrix

| Technology | Domain | Role & Purpose | Key Modules |
| :--- | :--- | :--- | :--- |
| **Python 3.10+** | Core Language | Primary runtime environment, scientific computing host, and glue architecture. | Entire repository |
| **PyTorch (torch, torchvision)** | Deep Learning | ResNet-1D and Spectrogram CNN-2D neural network modeling, GPU tensor acceleration, inference. | `astra_modulation_v2/`, `astra_modulation_2d/`, `astra_fusion/` |
| **NumPy** | Vectorized Math | High-performance array operations, matrix manipulation, bit-level packing/unpacking, and vectorized Viterbi trellis evaluation. | `astra_fec/`, `astra_synchronization/`, `astra_demodulation/`, `astra_synthetic/` |
| **SciPy (scipy.signal, scipy.fft)** | DSP & Numerical Analysis | STFT, Welch PSD, polyphase rational resampling, matched root-raised cosine (RRC) filtering, Hilbert transforms, and cyclic autocorrelation. | `astra_symbol_rate/`, `astra_synchronization/`, `astra_synthetic/` |
| **XGBoost (xgboost)** | Machine Learning | Gradient Boosted Decision Trees for multi-detector baud rate candidate ranking and end-to-end pipeline candidate scoring. | `astra_symbol_rate/ranker.py`, `astra_pipeline_scorer/` |
| **scikit-learn (sklearn)** | Data Science | Cross-validation utilities, confusion matrix generation, standard scalers, and metric tracking. | `scripts/`, `astra_modulation_v2/` |
| **Joblib** | Serialization | High-speed serialization and persistence for gradient boosted decision tree models and preprocessing pipelines. | `astra_symbol_rate/`, `models/` |
| **PyQt6 / PySide6** | Desktop GUI | Modern, responsive, cross-platform user interface for real-time waterfall inspection, constellation diagnostics, and payload exploration. | `astra_gui/` |
| **PyQtGraph / OpenGL** | GPU Visualization | 60 FPS real-time rendering of complex IQ constellations, 3D signal trajectories, waterfall spectrograms, and eye diagrams. | `astra_gui/src/world3d/`, `astra_gui/src/views/` |
| **PyYAML (yaml)** | Configuration | Declarative configuration management for demodulators, synchronizers, candidate grids, and pipeline profiles. | `configs/`, `astra_config/` |
| **CUDA (NVIDIA)** | Hardware Acceleration | GPU acceleration for PyTorch deep neural networks, batch STFT processing, and parallel candidate hypothesis testing. | `astra_modulation_v2/`, `astra_modulation_2d/` |

---

## Detailed Component & Module Integration

### 1. PyTorch Deep Learning
- **Why Chosen:** Provides dynamic execution graphs, native complex tensor support, and seamless GPU offloading with low inference latency.
- **Where Used:**
  - `astra_modulation_v2/models/resnet1d.py`: Implements deep 1D residual blocks extracting time-domain modulation features directly from raw 2,048-length normalized I/Q vectors.
  - `astra_modulation_2d/models/cnn2d.py`: Evaluates time-frequency textures on STFT spectrogram images.
  - `astra_fusion/fusion.py`: Merges multi-model logits with calibrated confidence weighting.

### 2. Vectorized DSP with NumPy & SciPy
- **Why Chosen:** Real-time processing demands sub-millisecond execution for signal conditioning, phase locking, and trellis decoding. Pure Python loops are replaced with compiled vector primitives.
- **Where Used:**
  - `astra_synchronization/sync_v2.py`: Rational polyphase resampling to 2 SPS, 2-state Gardner timing error detectors, and decision-directed Costas loops.
  - `astra_fec/viterbi.py`: Vectorized NumPy trellis processor computing branch and path metrics across all 64 states simultaneously ($< 30\text{ ms}$ decode time).
  - `astra_symbol_rate/cyclostationary.py`: Fast Cyclic Autocorrelation Function (CAF) computation using parallel FFT multiplication.

### 3. Gradient Boosted Decision Trees (XGBoost)
- **Why Chosen:** Provides highly non-linear, interpretable ranking over heterogeneous DSP feature vectors (spectral peak ratios, EVM, SNR, cyclostationary harmonics) without the overfitting risks of deep networks on tabular metadata.
- **Where Used:**
  - `astra_symbol_rate/ranker.py`: Classifies whether an estimated baud rate is a fundamental rate or a subharmonic/harmonic trap.
  - `astra_pipeline_scorer/scorer.py`: Combines Stage 6–10 metrics into a unified recovery confidence score.

### 4. PyQt6 / PySide6 & PyQtGraph Interactive UI
- **Why Chosen:** Native OS performance, thread-safe background execution for long DSP tasks, and OpenGL-accelerated plotting capable of rendering millions of IQ points without UI freezing.
- **Where Used:**
  - `astra_gui/src/app/main_window.py`: Central multi-tab dashboard.
  - `astra_gui/src/views/waterfall.py`: Real-time scrolling spectral waterfall.
  - `astra_gui/src/views/constellation.py`: Interactive persistence constellation with EVM overlays.
  - `astra_payload_explorer/explorer.py`: Hex/ASCII bitstream inspector and frame carver.

---

## Software Licensing & Third-Party Dependencies

All dependencies utilized by ASTRA are open-source and permissible under MIT, BSD-3, or Apache-2.0 licenses.
