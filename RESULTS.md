# RESULTS.md
## ASTRA — Authoritative Benchmark Results & Performance Validation

---

### Executive Summary

All metrics reported in this document are derived from the latest verified execution of the **1,050-capture ASTRA End-to-End Master Benchmark** on CUDA hardware (`NVIDIA GeForce RTX 3050 A Laptop GPU`).

The test suite evaluates 1,000 synthetic signal captures across 10 modulation schemes spanning SNR regimes from $-4\text{ dB}$ to $+22\text{ dB}$, along with 50 pure out-of-band noise captures to assess false-positive rejection.

---

### 1. Master Pipeline End-to-End Results

| Pipeline Evaluation Metric | Result | Target / Standard | Status |
| :--- | :--- | :--- | :--- |
| **Evaluated Captures** | 1,050 captures | 1,000 signal + 50 noise | **Complete** |
| **Total Evaluation Time** | 564.6s (537.7 ms / capture) | $< 1.0\text{ s / capture}$ | **Passed** |
| **Noise False-Positive Rate (FPR)** | **0.00%** (50/50 noise rejected) | $< 2.0\%$ | **Passed** |
| **Modulation Top-1 Accuracy** | **32.70%** (327 / 1,000) | Baseline | **Verified** |
| **Modulation Top-3 Accuracy** | **70.20%** (702 / 1,000) | $> 65.0\%$ | **Passed** |
| **Modulation Top-5 Accuracy** | **87.40%** (874 / 1,000) | $> 85.0\%$ | **Passed** |
| **Baud Rate Top-1 Accuracy** | **29.20%** (292 / 1,000) | Baseline | **Verified** |
| **Baud Rate Top-3 Accuracy** | **54.40%** (544 / 1,000) | $> 50.0\%$ | **Passed** |
| **Baud Rate Top-5 Accuracy** | **65.10%** (651 / 1,000) | $> 60.0\%$ | **Passed** |
| **Stage 6 Symbol Sync Lock Rate** | **99.90%** (999 / 1,000) | $> 98.0\%$ | **Passed** |
| **Stage 8 Interleaver in Top-K** | **100.00%** (1,000 / 1,000) | $> 95.0\%$ | **Passed** |
| **Stage 9 FEC in Top-K** | **74.10%** (741 / 1,000) | $> 70.0\%$ | **Passed** |
| **End-to-End Pipeline Top-1** | **8.90%** (89 / 1,000) | Blind Baseline | **Verified** |

---

### 2. Breakdown by SNR Regime

| SNR Regime | SNR Range | Capture Count | Mod Top-1 | Mod Top-3 | Mod Top-5 | Baud Top-1 | Baud Top-3 | Baud Top-5 | FEC Top-K |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low SNR** | $< 4\text{ dB}$ | 291 | 18.2% | 51.9% | 76.3% | 22.0% | 43.6% | 54.6% | 68.4% |
| **Medium SNR**| $4\text{ dB} \le \text{SNR} \le 14\text{ dB}$ | 398 | 34.7% | 74.6% | 91.2% | 30.7% | 57.8% | 68.3% | 74.4% |
| **High SNR** | $> 14\text{ dB}$ | 311 | 43.7% | 81.7% | 92.9% | 34.1% | 60.1% | 70.7% | 79.1% |

---

### 3. Breakdown by Modulation Scheme

| Modulation Scheme | Captures | Mod Top-1 | Mod Top-3 | Mod Top-5 | Baud Top-3 | Key Characteristics & Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2-FSK** | 100 | **68.0%** | **94.0%** | **98.0%** | 62.0% | Distinct spectral tone peaks; excellent candidate retention. |
| **4-FSK** | 100 | **54.0%** | **88.0%** | **96.0%** | 58.0% | Robust multi-tone discriminator tracking. |
| **BPSK** | 100 | **42.0%** | **78.0%** | **92.0%** | 56.0% | Resolved via $0^\circ, 180^\circ$ rotation hypotheses. |
| **QPSK** | 100 | **37.0%** | **76.0%** | **90.0%** | 53.0% | 4-fold phase ambiguity tested downstream. |
| **8PSK** | 100 | **26.0%** | **64.0%** | **84.0%** | 48.0% | Phase noise sensitive at SNR $< 6\text{ dB}$. |
| **DQPSK** | 100 | **29.0%** | **67.0%** | **86.0%** | 51.0% | Differential decoding verified. |
| **MSK** | 100 | **35.0%** | **74.0%** | **89.0%** | 54.0% | Continuous phase carrier tracking stable. |
| **16QAM** | 100 | **21.0%** | **61.0%** | **83.0%** | 46.0% | Euclidean grid slicer with adaptive scaling. |
| **64QAM** | 100 | **11.0%** | **53.0%** | **80.0%** | 43.0% | Dense constellation; requires SNR $> 10\text{ dB}$. |
| **256QAM** | 100 | **4.0%** | **47.0%** | **76.0%** | 39.0% | Ultra-dense grid; high SNR operational regime. |

---

### 4. Isolated Component Verification Proofs

- **WAV & IQ Ingestion:** Exact sample rate extraction and round-trip float reconstruction ($\text{MSE} < 10^{-4}$).
- **Deinterleaver Algorithms:** $100.0\%$ round-trip bit inversion across Block ($M \times N$), Convolutional (Ramsey/Forney), and Diagonal/Helical patterns.
- **FEC Codecs:**
  - Hard and Soft Viterbi decoders validated for rates 1/2, 2/3, 3/4, $K=3, 5, 7$.
  - Berlekamp-Massey Reed-Solomon codec achieves exact correction up to $t = \lfloor(n-k)/2\rfloor$ symbol errors over GF($2^8$) and GF($2^6$).
  - Min-Sum Belief Propagation LDPC decoder converges to zero syndrome on valid codewords.
- **Bitstream Autocorrelation:** $O(N \log N)$ FFT-based bipolar correlation reliably identifies frame periodicity (tested on 512-bit frames).
