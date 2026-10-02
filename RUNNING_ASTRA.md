# Running ASTRA: Execution Manual & User Guide

This document contains exact, step-by-step instructions for installing, configuring, and executing ASTRA across different operational modes: Standalone Demo, Interactive Desktop GUI, Command Line Interface (CLI), and Full Automated Benchmark Suite.

---

## 1. System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python Version:** Python 3.10 to 3.12 (Recommended: 3.10)
- **RAM:** Minimum 8 GB (16 GB Recommended)
- **GPU (Optional but Recommended):** NVIDIA GPU with CUDA 11.8+ (Automatic CPU fallback supported)

---

## 2. Installation & Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/sriramjayanti/ASTRA.git
cd ASTRA

# 2. Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install all production dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verify System Health

Run the diagnostic validation script to verify dependencies, CUDA support, model checkpoints, and pipeline imports:

```bash
python scripts/validate_installation.py
```

Expected output:
```text
============================================================
  ASTRA SYSTEM HEALTH & INSTALLATION VALIDATION
============================================================
[*] Python Version: 3.10.x
[+] Python version compatible: PASS
...
  ASTRA SYSTEM HEALTH CHECK: PASSED (System Ready)
============================================================
```

---

## 4. Operational Modes

### Mode A: Interactive Desktop Workstation & 3D Signal World (GUI)
Launch the full interactive 3D signal intelligence workstation:

```bash
# Recommended launcher:
python astra_gui/main.py

# Or as module:
python -m astra_gui.main
```

**Workstation Features:**
- **Auto Screen Adaptation:** Dynamically fits within 96% of your monitor's available resolution and centers cleanly.
- **Dampened 3D Navigation:** Smooth orbit/pan without sudden jumping or camera clipping.
- **Full-Screen Scientific Dock:** Click the **`⛶ FULL SCREEN`** button in the bottom dock tab bar to expand Time Domain, FFT/PSD, Waterfall, Constellation, Eye Diagram, Bitstream, and Hex/Payload viewers to 100% screen height; click **`🗗 RESTORE VIEW`** to return to split mode.
- **Instant Demo:** Click `RUN DEMO` in the top bar to run a complete blind signal extraction demonstration directly in the UI.

---

### Mode B: 3-Minute Standalone Judge Demo
Executes an end-to-end synthetic IQ signal ingestion, modulation classification, baud estimation, synchronization, demodulation, Viterbi decoding, and payload recovery without requiring external datasets:

```bash
python scripts/run_demo.py
```

---

### Mode C: Complete 16-Stage End-to-End Master Pipeline
Executes all 16 AI models and DSP stages sequentially on an arbitrary RF capture:

```bash
python run_complete_astra_ai.py
```

---

### Mode D: Command Line Interface (CLI) Analysis
Analyze any raw `.wav` or `.iq` recording directly from the terminal:

```bash
# Single file analysis with automatic candidate beam search
python -m astra_cli.main --input test_signals/sample_qpsk.wav --output-dir outputs/analysis_report/

# With explicit sample rate override (e.g., 2 MSps):
python -m astra_cli.main --input test_signals/sample_qpsk.wav --sample-rate 2000000 --output-dir outputs/
```

Key CLI Output Artifacts generated in `--output-dir`:
- `summary.json`: Top modulation, baud rate, EVM, SNR, and validated payload.
- `constellation.png`: Synchronized complex I/Q scatter plot.
- `waterfall.png`: Power spectral density and spectrogram visualization.
- `payload.bin` / `payload.txt`: Extracted, deinterleaved, and FEC-corrected bitstream.

---

### Mode E: Full 70-Signal Verification Suite
Evaluates all multi-standard test signals across diverse SNR tiers:

```bash
python -u scripts/run_full_verification_suite.py
```

---

## 5. Running Automated Unit Tests

Run the complete Pytest suite to verify DSP modules, FEC algorithms, and demodulators:

```bash
pytest tests/ -v
```
