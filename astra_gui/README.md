# ASTRA STAGE 16: Immersive Desktop GUI + 3D Signal Processing Experience

## 1. Overview & Vision
The **ASTRA Desktop Workstation (`astra_gui`)** represents the final crown of the ASTRA automated signal intelligence framework. Designed as a scientific command center, advanced signal laboratory, and visual explanation system, it delivers an immersive **3D Signal World** that lets users watch an unknown RF signal transform from raw electromagnetic energy into verified, recovered payload data.

```
UNKNOWN RF SIGNAL
  ↓
ANALYZED WAVEFORM (3D Waveform Tunnel)
  ↓
MODULATION HYPOTHESES (Neural Branching Universe)
  ↓
SYMBOL RATE (Baud Spectral Peaks)
  ↓
SYNCHRONIZATION (Costas Loop CFO Stabilization Chamber)
  ↓
CONSTELLATION LOCK (3D Point Cloud & Cluster Centroids)
  ↓
DEMODULATION (Decision Planes & Bit Slicing)
  ↓
DEINTERLEAVING (16x16 Matrix Reordering)
  ↓
FEC CORRECTION (Viterbi Trellis & Codeword Repair)
  ↓
VALIDATION GATES (CRC-32, Parity, Periodicity)
  ↓
PIPELINE RANKING (XGBoost Hypothesis Race)
  ↓
BITSTREAM STRUCTURE (Autocorrelation Periodicity Ribbon)
  ↓
TRANSFORMER SEGMENTATION (Sync, Header, Payload, CRC Regions)
  ↓
HEADER / PAYLOAD RECOVERY (Hex Matrix & UTF-8 Text Emergence)
  ↓
FINAL EXPLAINABILITY (3D Evidence DAG & Canonical Statuses)
```

---

## 2. Core Architecture & Stack
- **UI Framework:** PySide6 (Qt 6 for Python) with High-DPI core profile scaling.
- **Scientific 2D Plotting:** PyQtGraph for ultra-low latency real-time time-series, FFT/PSD, STFT waterfall, and 2D constellation scatter plots.
- **3D Signal World:** PyQtGraph OpenGL (`GLViewWidget`) with 60 FPS animation loop, orbital camera controller, and shader lighting.
- **Event-Driven Architecture:** `VisualizationEventBus` connecting background DSP/ML worker threads with the GUI without thread locking.
- **Large Signal LOD/Decimation:** Multi-resolution decimation ensuring that massive 100M+ sample captures render smoothly at 60 FPS without memory exhaustion.

---

## 3. Workstation Layout & The 6 Primary UI Zones

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ZONE A: TOP COMMAND BAR (Branding, Capture Info, AUTO/EXPERT, Analyze, Demo)│
├───────────────┬─────────────────────────────────────────────┬───────────────┤
│               │                                             │               │
│ ZONE B:       │                                             │ ZONE D:       │
│ PIPELINE      │           ZONE C: 3D SIGNAL WORLD           │ EVIDENCE      │
│ NAVIGATOR     │                                             │ INSPECTOR     │
│ (Stages 1-15) │  (Interactive 3D Waveforms, Constellations, │ (Status Badge,│
│               │   Bit Tunnels, Trellis Repair, Evidence)   │  Confidence,  │
│               │                                             │  Support/Con) │
│               │                                             │               │
├───────────────┴─────────────────────────────────────────────┴───────────────┤
│ ZONE E: SIGNAL JOURNEY TIMELINE (RAW ━ DSP ━ MOD ━ SYNC ━ FEC ━ PAYLOAD)    │
├─────────────────────────────────────────────────────────────────────────────┤
│ ZONE F: SCIENTIFIC DOCK (Time, FFT, Waterfall, Eye, Bits, Frames, Hex, Logs)│
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Zone A (Top Command Bar):** Title, capture name, sampling rate, duration, AUTO/EXPERT mode toggle, ANALYZE trigger, and RUN DEMO button.
- **Zone B (Left Pipeline Navigator):** Displays all 15 stages with real-time execution indicators (`WAITING`, `RUNNING`, `DONE`, `WARNING`, `FAILED`).
- **Zone C (Central 3D Signal World):** 15 dedicated OpenGL scenes visualizing physical and mathematical transformations with smooth orbital camera controls.
- **Zone D (Right Evidence Inspector):** Displays localized confidence scores, canonical status badges (`CONFIRMED`, `ESTIMATED`, `POSSIBLE`, `UNKNOWN`), supporting evidence, contradictions, and competing candidate paths.
- **Zone E (Signal Journey Timeline):** Interactive scrubber with play/pause, step forward/backward, replay, and speed selection (`0.25x` to `Instant`).
- **Zone F (Bottom Scientific Dock):** High-precision tabs for Time-Domain (IQ), Spectrum/PSD, 2D Waterfall Spectrogram, 2D Constellation, Eye Diagram, Bitstream Octets, Frame Tables, Hex/ASCII Dump, Confidence Breakdown, Candidate Tree, and System Logs.

---

## 4. The 15 Specialized 3D Signal Scenes

1. **Stage 1 (Raw Signal Scene):** Helical 3D IQ waveform tunnel $(X: \text{time}, Y: I, Z: Q)$ and streaming sample particles.
2. **Stage 2 (DSP Preprocessing Scene):** 3D frequency $\times$ time $\times$ power terrain surface and matched filter response curve.
3. **Stage 3 (Modulation Discovery Scene):** Candidate modulation worlds (QPSK, 8PSK, 16QAM) with confidence branches.
4. **Stage 4 (Symbol Rate Scene):** Spectral pulse lanes (4800, 9600, 19200 baud) with synchronized temporal rhythms.
5. **Stage 5 (Candidate Universe Scene):** 3D orbital rings showing candidate hypotheses arranged by ranking score.
6. **Stage 6 (Synchronization Scene):** Constellation rotation slowing from $+1186\text{ Hz}$ to $+12\text{ Hz}$ residual error as Costas loop locks.
7. **Stage 7 (Constellation Scene):** 3D constellation cloud with 4 compact QPSK clusters and centroid markers.
8. **Stage 8 (Deinterleaving Scene):** $16 \times 16$ grid of 256 bit tiles scrambling and resolving into sequential order.
9. **Stage 9 (FEC Error Correction Scene):** Corrupted bit positions highlighted in red transforming into emerald-green repaired bits.
10. **Stage 10 (Validation Gates Scene):** Physical 3D gates (SYNC, FEC, FRAME, CRC) illuminating green on 8/8 CRC verification.
11. **Stage 11 (Pipeline Ranking Scene):** Winner candidate pipeline advancing forward while weaker paths fade.
12. **Stage 12 (Bitstream Structure Scene):** 3D bit ribbon with recurring 512-bit autocorrelation pulses.
13. **Stage 13 (CNN+Transformer Scene):** Functional color-coded frame regions: SYNC (Cyan), HEADER (Violet), PAYLOAD (Green), CRC (Magenta).
14. **Stage 14 (Payload Vault Scene):** Recovered payload byte cubes displaying hex matrix (`68 69 20 68 65 6C 6C 6F`) and ASCII text ('hi hello').
15. **Stage 15 (Explainability Scene):** 3D reasoning network with central confirmed claim orbited by supporting/contradicting evidence.

---

## 5. Execution Instructions

### Running the Workstation
```bash
# Launch ASTRA Desktop Workstation
python -m astra_gui.main
# or
python astra_gui/main.py
```

### Running the Full Test Suite
```bash
.venv\Scripts\python -m pytest astra_gui/tests/ -v
# Output: 12 passed in 8.95s (100% passing)
```
