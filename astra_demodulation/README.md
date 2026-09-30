# ASTRA Stage 7 — Demodulation Engine

## Overview
The **Demodulation Engine** is Stage 7 of the **ASTRA (Automated Signal Analysis & Recovery Assistant)** signal processing pipeline.

Following Stage 6 Synchronization (CFO correction, matched filtering, symbol timing, and carrier recovery), Stage 7 converts the synchronized complex symbol points into:
1. **Hard Decision Slices** (Integer symbol indices and Gray-coded bitstream).
2. **True Soft Decision Log-Likelihood Ratios (LLRs)** ($P(b_k=0|y) / P(b_k=1|y)$) required by downstream soft-input FEC decoders (Viterbi, LDPC).
3. **Rotational Phase-Ambiguity Bitstream Variants** ($0^\circ, 90^\circ, 180^\circ, 270^\circ$ for QPSK/QAM, and $0^\circ, 45^\circ, \dots, 315^\circ$ for 8PSK) to ensure frame and CRC synchronization succeed regardless of carrier lock orientation.
4. **Demodulation Quality & EVM Telemetry** (Error Vector Magnitude in %, dB, RMS, noise variance $\sigma^2$, and mean decision margins).

---

## Supported Modulation Schemes & Canonical Gray Mappings

All digital mappings strictly follow MSB-first bit ordering and unit average constellation energy ($E_s = 1.0$):

| Modulation | Bits/Symbol | Decision Regions | Ambiguity Rotations | LLR Modes |
| :--- | :---: | :--- | :--- | :--- |
| **BPSK** | 1 | Real axis sign slicer ($\ge 0 \to 0, < 0 \to 1$) | $[0^\circ, 180^\circ]$ | Exact, Max-Log |
| **QPSK** | 2 | Gray quadrant mapping (`00`, `01`, `11`, `10`) | $[0^\circ, 90^\circ, 180^\circ, 270^\circ]$ | Exact, Max-Log |
| **8PSK** | 3 | Circular Gray mapping (8 points, $\pi/4$ steps) | $[0^\circ, 45^\circ, 90^\circ, \dots, 315^\circ]$ | Exact, Max-Log |
| **16-QAM** | 4 | Square Gray grid ($\{-3, -1, 1, 3\}/\sqrt{10}$) | $[0^\circ, 90^\circ, 180^\circ, 270^\circ]$ | Exact, Max-Log |
| **64-QAM** | 6 | Square Gray grid ($\{-7, -5, \dots, 7\}/\sqrt{42}$) | $[0^\circ, 90^\circ, 180^\circ, 270^\circ]$ | Exact, Max-Log |
| **2-FSK** | 1 | Tone frequency discriminator / Energy threshold | Normal, Swapped Tone | Soft Tone Margin |
| **4-FSK** | 2 | 4-tone Gray frequency slicing | Normal, Swapped Tone | Grouped Tone LLR |

---

## ASTRA LLR Sign Convention

$$\text{LLR}(b_k) = \ln \left( \frac{P(b_k = 0 \mid y)}{P(b_k = 1 \mid y)} \right)$$

- $\text{LLR} > 0 \implies$ **Bit 0** is favored.
- $\text{LLR} < 0 \implies$ **Bit 1** is favored.
- $|\text{LLR}| \gg 0 \implies$ High confidence decision.
- $|\text{LLR}| \approx 0 \implies$ Ambiguous bit on decision boundary.

---

## Directory Structure

```
astra_demodulation/
├── configs/
│   └── demodulation_config.yaml       # Configuration parameters and thresholds
├── src/
│   ├── __init__.py                    # Package exports
│   ├── models.py                      # DemodulationResult, DemodulationVariant, Quality
│   ├── mappings.py                    # Canonical Gray bit mapping definitions
│   ├── constellation.py               # Constellation definition utilities & subsets
│   ├── hard_decision.py               # Vectorized nearest-neighbor slicer
│   ├── llr.py                         # Exact (logsumexp) and Max-Log LLR engine
│   ├── psk.py                         # BPSK, QPSK, 8PSK demodulators
│   ├── qam.py                         # 16-QAM, 64-QAM demodulators
│   ├── fsk.py                         # 2-FSK, 4-FSK demodulators
│   ├── ambiguity.py                   # Phase ambiguity variant generator
│   ├── noise.py                       # Noise variance estimator
│   ├── evm.py                         # Error Vector Magnitude (EVM) calculator
│   ├── quality.py                     # Demodulation quality telemetry
│   ├── validators.py                  # Input bounds and NaN/Inf validation
│   ├── router.py                      # Demodulation router
│   ├── inference.py                   # DemodulationEngine main API
│   └── utils.py                       # Synthetic test data generators
├── tests/
│   ├── test_demodulation.py           # Master 30-test unit suite
│   ├── test_bpsk.py                   # BPSK unit tests
│   ├── test_qpsk.py                   # QPSK unit tests
│   ├── test_8psk.py                   # 8PSK unit tests
│   ├── test_16qam.py                  # 16-QAM unit tests
│   ├── test_64qam.py                  # 64-QAM unit tests
│   ├── test_2fsk.py                   # 2-FSK unit tests
│   ├── test_4fsk.py                   # 4-FSK unit tests
│   ├── test_llr.py                    # LLR unit tests
│   ├── test_ambiguity.py              # Ambiguity rotation tests
│   ├── test_evm.py                    # EVM unit tests
│   └── test_end_to_end_demodulation.py# Stage 6 -> Stage 7 integration tests
├── examples/
│   └── demodulate_candidate.py        # End-to-end demonstration script
├── run_tests.py                       # Test runner
├── requirements.txt                   # Python dependencies
└── README.md                          # Documentation
```

---

## Quickstart

```python
from astra_synchronization.src.inference import SynchronizationEngine
from astra_demodulation.src.inference import DemodulationEngine

# 1. Run Synchronization (Stage 6)
sync_engine = SynchronizationEngine()
sync_result = sync_engine.synchronize(rx_iq, hypothesis)

# 2. Run Demodulation (Stage 7)
demod_engine = DemodulationEngine()
demod_result = demod_engine.demodulate(sync_result, hypothesis)

print(f"Status      : {demod_result.status}")
print(f"Hard Bits   : {len(demod_result.hard_bits)} bits")
print(f"EVM         : {demod_result.quality.evm_percent:.2f}% ({demod_result.quality.evm_db:.2f} dB)")
print(f"Variants    : {len(demod_result.phase_variants)} rotational bitstreams generated")
```

---

## Verification & Testing

Run the full unit test suite:
```bash
python astra_demodulation/run_tests.py
```
**Result:** 30/30 unit tests pass (100% success rate).
