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
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) Install GUI dependencies if using Desktop UI
pip install PyQt6 pyqtgraph PyOpenGL
```

---

## 3. Verify System Health

Run the diagnostic validation script to verify dependencies, CUDA support, model checkpoints, and pipeline imports:

```bash
python scripts/validate_installation.py
```

Expected output:
```
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

### Mode A: 3-Minute Standalone Judge Demo
Executes an end-to-end synthetic IQ signal ingestion, modulation classification, baud estimation, synchronization, demodulation, Viterbi decoding, and payload recovery without requiring external datasets:

```bash
python scripts/run_demo.py
```

---

### Mode B: Command Line Interface (CLI) Analysis
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

### Mode C: Interactive Desktop GUI (PyQt6)
Launch the full interactive signal triage environment:

```bash
python -m astra_gui.src.app.main_window
```

**GUI Workflow:**
1. **Load Capture:** Click `File` $\rightarrow$ `Open Signal Capture...` and select any `.wav`, `.iq`, or `.sigmf` file.
2. **Execute Triage:** Click `Run Complete Analysis` or press `F5`.
3. **Inspect Views:**
   - **Spectrum Tab:** Live FFT, Welch PSD, and interactive 2D Waterfall.
   - **Modulation & Baud Tab:** Top-5 neural network confidence scores and cyclostationary baud peaks.
   - **Synchronization Tab:** Costas loop phase trajectory, Gardner timing jitter, and fine CFO correction.
   - **Constellation Tab:** High-density persistence plot with EVM and IQ imbalance metrics.
   - **Bitstream & Framing Tab:** Raw demodulated bits, deinterleaved matrix, FEC syndromes, and frame preamble markers.
   - **Payload Explorer:** Live ASCII, UTF-8, and Hex carver with CRC status indicators.

---

### Mode D: Automated Master Benchmark
Run the comprehensive 1,050-capture end-to-end evaluation suite across all 10 modulation classes and SNR ranges:

```bash
python scripts/benchmark_end_to_end_astra.py --num-captures 1050 --output docs/archive/MASTER_BENCHMARK_RESULTS.md
```

---

## 5. Running Automated Unit Tests

Run the complete Pytest suite to verify DSP modules, FEC algorithms, and demodulators:

```bash
pytest tests/ -v
```
