# ASTRA: Setup & Execution Guide

This document outlines the step-by-step instructions to install, configure, and execute the ASTRA pipeline, synthetic signal generator, CLI tools, and interactive GUI.

---

## 1. System Requirements

- **OS:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python:** Version 3.10, 3.11, or 3.12
- **RAM:** Minimum 8 GB (16 GB recommended)
- **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ for accelerated neural network inference (CPU execution is fully supported).

---

## 2. Environment Setup

### 2.1 Clone the Repository
```bash
git clone https://github.com/sriramjayanti/ASTRA.git
cd ASTRA
```

### 2.2 Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Quick Verification & Isolated Stage Tests

Run the isolated downstream stage verification script to ensure interleaver permutations and FEC decoders are working:

```bash
python scripts/test_downstream_stages_isolated.py
```

Expected output:
```
======================================================================
   ASTRA DOWNSTREAM ISOLATED ROUND-TRIP UNIT VERIFICATION
======================================================================
[PASS] Interleaver NONE exact round-trip: 0 diffs
[PASS] Interleaver BLOCK 16x16 exact round-trip: 0 diffs
[PASS] Interleaver CONVOLUTIONAL exact round-trip: 0 diffs
[PASS] Interleaver HELICAL exact round-trip: 0 diffs
[PASS] Interleaver PSEUDO-RANDOM exact round-trip: 0 diffs
[PASS] FEC NO_FEC exact round-trip: 0 diffs
[PASS] FEC Convolutional Viterbi K=7 exact round-trip: 0 diffs
[PASS] FEC Reed-Solomon (255,223) exact round-trip: 0 diffs
[PASS] FEC LDPC exact round-trip: 0 diffs
======================================================================
ALL ISOLATED DOWNSTREAM ROUND-TRIP TESTS PASSED (100.0% SUCCESS)
======================================================================
```

---

## 4. Running the Full Blind Pipeline

To run the complete 14-stage blind inference pipeline on an I/Q file or synthetic test capture:

```bash
python run_complete_astra_ai.py
```

### Command-Line Arguments:
```bash
python run_complete_astra_ai.py --help
```

Options:
- `--input <path>`: Path to `.iq` (complex64) or `.wav` input recording.
- `--sample-rate <float>`: Sampling rate in Hz (default: 192,000 Hz).
- `--output-dir <path>`: Directory to store extraction reports and lineage JSONs.
- `--verbose`: Enable detailed per-stage debug telemetry logging.

---

## 5. Generating Synthetic Signal Datasets

ASTRA includes a built-in synthetic signal generation engine (`astra_synthetic`) that simulates realistic channel impairments (CFO, multipath Rayleigh fading, timing offsets, and AWGN):

```bash
python generate_custom_payload_signals.py
```

To generate unseen multi-class blind evaluation sets:
```bash
python generate_unseen_signals.py
```

---

## 6. Launching the Interactive GUI

ASTRA includes a modern desktop interface built with PySide6/Qt featuring 3D constellation visualizers, live waterfall spectral analyzers, and a bitstream explorer:

```bash
python -m astra_gui.src.main
```

Features:
1. **Live Spectral View:** Real-time FFT PSD and waterfall display.
2. **3D Constellation Sphere:** Interactive OpenGL point cloud for spatial cluster analysis.
3. **Synchronization Monitor:** Real-time lock indicator for Gardner TED and Costas carrier loops.
4. **Hex & Bitstream Inspector:** Live bit-level and ASCII payload extraction.

---

## 7. Running the Pytest Test Suite

Execute the automated test suite across all modules:

```bash
pytest tests/ -v
```

---

## 8. Troubleshooting & FAQ

- **CUDA / PyTorch warning on CPU:** If you do not have a dedicated GPU, ASTRA automatically defaults to CPU inference without any manual configuration.
- **Missing Checkpoint Files:** Ensure the `checkpoints/` directory is present in the repository root containing `astra_resnet1d_v2.pt` and `astra_spectrogram_cnn_v2.pt`.
