# ASTRA Master End-to-End System Benchmark Report
**Dataset:** `ASTRA_FINAL_TEST_SET`  
**Execution Timestamp:** 2026-09-30 17:50:02 UTC  
**Hardware Engine:** NVIDIA GeForce RTX 3050 A Laptop GPU (`cuda`)  
**Total Independent Captures:** 1050 (1,000 Communications + 50 Non-Target Noise)

---

## 1. Executive Benchmark Summary

This report establishes the first **unified, untouched end-to-end benchmark** for the entire ASTRA signal recovery pipeline. All 1,050 captures were evaluated continuously from raw IQ samples through modulation classification, symbol rate ranking, carrier/timing synchronization, demodulation, deinterleaving, forward error correction decoding, candidate ranking, and validation.

> [!IMPORTANT]
> **Zero Data Leakage Guarantee:**
> All 1,050 captures in `ASTRA_FINAL_TEST_SET` were synthesized with isolated seed `88888888` and were **never used** for model training, validation, threshold tuning, calibration, or model selection.

| Metric | Result | Target Benchmark |
| :--- | :--- | :--- |
| **Modulation Top-1** | **14.90%** | $\ge 50\%$ |
| **Modulation Top-3** | **36.00%** | $\ge 85\%$ |
| **Baud Top-1** | **21.20%** | $\ge 70\%$ |
| **Baud Top-3** | **45.80%** | $\ge 85\%$ |
| **Sync success** | **100.00%** | $\ge 60\%$ |
| **Correct demodulation** | **0.90%** | $\ge 55\%$ |
| **Correct interleaver in Top-K** | **100.00%** | $\ge 80\%$ |
| **Correct FEC in Top-K** | **74.10%** | $\ge 80\%$ |
| **Pipeline Top-1** | **3.00%** | $\ge 50\%$ |
| **Payload exact recovery** | **0.00%** | $\ge 35\%$ |
| **Noise false-positive rate** | **0.00%** | $\le 5\%$ |

---

## 2. SNR Tier Stratification Breakdown

| SNR Tier | Total Signals | Modulation Top-1 | Baud Top-1 | Sync Success | Demod Success | Payload Exact Recovery |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low (< 4 dB)** | 324 | 12.0% | 11.7% | 100.0% | 0.3% | 0.0% |
| **Medium (4-14 dB)** | 338 | 17.8% | 24.0% | 100.0% | 0.9% | 0.0% |
| **High (> 14 dB)** | 338 | 14.8% | 27.5% | 100.0% | 1.5% | 0.0% |

---

## 3. Modulation Family Breakdown

| Modulation Scheme | Captures | Top-1 Accuracy | Top-3 Accuracy | Baud Top-1 | Sync Success | Demod Success | Payload Exact Recovery |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2-FSK** | 100 | 24.0% | 79.0% | 15.0% | 100.0% | 1.0% | 0.0% |
| **4-FSK** | 100 | 12.0% | 34.0% | 4.0% | 100.0% | 0.0% | 0.0% |
| **BPSK** | 100 | 0.0% | 0.0% | 19.0% | 100.0% | 0.0% | 0.0% |
| **QPSK** | 100 | 3.0% | 29.0% | 28.0% | 100.0% | 0.0% | 0.0% |
| **8PSK** | 100 | 0.0% | 0.0% | 25.0% | 100.0% | 0.0% | 0.0% |
| **DQPSK** | 100 | 0.0% | 0.0% | 26.0% | 100.0% | 0.0% | 0.0% |
| **MSK** | 100 | 86.0% | 93.0% | 24.0% | 100.0% | 7.0% | 0.0% |
| **16QAM** | 100 | 0.0% | 3.0% | 26.0% | 100.0% | 1.0% | 0.0% |
| **64QAM** | 100 | 24.0% | 86.0% | 25.0% | 100.0% | 0.0% | 0.0% |
| **256QAM** | 100 | 0.0% | 36.0% | 20.0% | 100.0% | 0.0% | 0.0% |

---

## 4. Non-Target Noise & Interference Rejection

- **Total Non-Target Captures:** 50
  - 25 Pure AWGN Thermal Noise Captures
  - 15 Unmodulated CW Tone Captures
  - 10 Multi-Tone Continuous Interference Captures
- **False-Positive Triggers:** 0
- **Noise False-Positive Rate:** **0.00%**

ASTRA correctly suppressed non-target RF emissions, rejecting noise without falsely asserting verified communications data.
