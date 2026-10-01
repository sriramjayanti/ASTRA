# ASTRA Interactive User & Judge Walkthrough Guide

This guide walks you through the step-by-step operation of ASTRA, illustrating both GUI and CLI evaluation workflows.

---

## 1. Quick Launch Checklist

Ensure the environment is ready:
```bash
# 1. Activate environment
.venv\Scripts\activate   # (or source .venv/bin/activate on Linux/macOS)

# 2. Run installation health check
python scripts/validate_installation.py

# 3. Execute quick demo
python scripts/run_demo.py
```

---

## 2. Interactive Graphical Interface (GUI) Walkthrough

Launch the desktop interface:
```bash
python -m astra_gui.src.app.main_window
```

### Step 1: Open a Signal Capture
- Click on **`File`** $\rightarrow$ **`Open Signal Capture...`** (or press `Ctrl+O`).
- Navigate to `test_signals/` and choose any capture file (e.g., `sample_qpsk.wav` or `sample_8psk.iq`).
- The system automatically detects file format, data precision (Float32, Int16), and initial sampling rate.

### Step 2: Run Autonomous Triage
- Click the large green **`Run Complete Analysis`** button on the top toolbar (or press `F5`).
- The pipeline executes Stages 1 through 14 asynchronously without blocking the UI.
- Progress bars show real-time stage transitions (Preprocessing $\rightarrow$ Neural Classification $\rightarrow$ Sync $\rightarrow$ Viterbi/RS Decoding $\rightarrow$ Payload Carving).

### Step 3: Inspect Multi-Domain Visualizations
- **Waterfall & Spectrum View:** Inspect the real-time FFT, center frequency offset, and scrolling STFT power spectrogram.
- **Candidate Beam View:** Review the Top-5 neural network modulation predictions and Top-3 cyclostationary baud rate estimations.
- **Constellation & Topology View:** View the synchronized IQ scatter plot. Observe real-time EVM (Error Vector Magnitude), SNR, and phase jitter metrics.
- **Bitstream & Frame Explorer:** View raw demodulated bits, bitstream entropy histograms, and discovered frame synchronization words (`0xEB90`, `0x1ACFFC1D`).
- **Recovered Payload:** Read the extracted application payload in plain text ASCII, UTF-8, or raw hexadecimal.

### Step 4: Export Forensic Report
- Click **`File`** $\rightarrow$ **`Export Forensic Report...`**
- A comprehensive JSON/PDF report containing complete signal lineage, parameters, and decoded bits is saved to your chosen destination.

---

## 3. Command Line Interface (CLI) Walkthrough

For headless servers, automated batch processing, or CI/CD pipelines:

```bash
# Basic triage of a single capture
python -m astra_cli.main --input test_signals/sample_qpsk.wav --output-dir outputs/

# Batch triage of an entire directory of unknown signals
python -m astra_cli.main --batch-dir test_signals/ --output-dir outputs/batch_run/ --top-k 5
```

The CLI outputs:
- Real-time ASCII progress bars.
- Colored log messages summarizing each stage.
- Structured `summary.json` containing confidence scores, EVM, and decoded payloads.
