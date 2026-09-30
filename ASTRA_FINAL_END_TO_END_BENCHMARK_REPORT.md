# ASTRA Master End-to-End System Benchmark Report
**Dataset:** `ASTRA_FINAL_TEST_SET`  
**Execution Timestamp:** 2026-09-30 18:54:38 UTC  
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
| **Modulation Top-1** | **32.70%** | $\ge 50\%$ |
| **Modulation Top-3** | **70.20%** | $\ge 85\%$ |
| **Baud Top-1** | **27.00%** | $\ge 70\%$ |
| **Baud Top-3** | **51.00%** | $\ge 85\%$ |
| **Sync success** | **99.90%** | $\ge 60\%$ |
| **Correct demodulation** | **0.40%** | $\ge 55\%$ |
| **Correct interleaver in Top-K** | **100.00%** | $\ge 80\%$ |
| **Correct FEC in Top-K** | **74.10%** | $\ge 80\%$ |
| **Pipeline Top-1** | **8.60%** | $\ge 50\%$ |
| **Payload exact recovery** | **0.00%** | $\ge 35\%$ |
| **Noise false-positive rate** | **0.00%** | $\le 5\%$ |

---

## 2. SNR Tier Stratification Breakdown

| SNR Tier | Total Signals | Modulation Top-1 | Baud Top-1 | Sync Success | Demod Success | Payload Exact Recovery |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low (< 4 dB)** | 324 | 25.6% | 16.0% | 100.0% | 0.0% | 0.0% |
| **Medium (4-14 dB)** | 338 | 32.0% | 30.5% | 99.7% | 0.6% | 0.0% |
| **High (> 14 dB)** | 338 | 40.2% | 34.0% | 100.0% | 0.6% | 0.0% |

---

## 3. Modulation Family Breakdown

| Modulation Scheme | Captures | Top-1 Accuracy | Top-3 Accuracy | Baud Top-1 | Sync Success | Demod Success | Payload Exact Recovery |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2-FSK** | 100 | 55.0% | 93.0% | 18.0% | 100.0% | 2.0% | 0.0% |
| **4-FSK** | 100 | 55.0% | 93.0% | 9.0% | 100.0% | 0.0% | 0.0% |
| **BPSK** | 100 | 49.0% | 62.0% | 33.0% | 100.0% | 0.0% | 0.0% |
| **QPSK** | 100 | 24.0% | 69.0% | 37.0% | 100.0% | 1.0% | 0.0% |
| **8PSK** | 100 | 11.0% | 38.0% | 32.0% | 100.0% | 0.0% | 0.0% |
| **DQPSK** | 100 | 13.0% | 60.0% | 34.0% | 99.0% | 0.0% | 0.0% |
| **MSK** | 100 | 62.0% | 95.0% | 23.0% | 100.0% | 1.0% | 0.0% |
| **16QAM** | 100 | 24.0% | 59.0% | 28.0% | 100.0% | 0.0% | 0.0% |
| **64QAM** | 100 | 15.0% | 63.0% | 31.0% | 100.0% | 0.0% | 0.0% |
| **256QAM** | 100 | 19.0% | 70.0% | 25.0% | 100.0% | 0.0% | 0.0% |

---

## 4. Non-Target Noise & Interference Rejection

- **Total Non-Target Captures:** 50
  - 25 Pure AWGN Thermal Noise Captures
  - 15 Unmodulated CW Tone Captures
  - 10 Multi-Tone Continuous Interference Captures
- **False-Positive Triggers:** 0
- **Noise False-Positive Rate:** **0.00%**

ASTRA correctly suppressed non-target RF emissions, rejecting noise without falsely asserting verified communications data.
