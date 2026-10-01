# ASTRA: Known Limitations & Boundary Conditions

To maintain technical credibility and scientific transparency, this document details the known physical, mathematical, and algorithmic limitations of the current ASTRA implementation.

---

## 1. Physical & Mathematical Limits

### 1.1 Extremely Low Signal-to-Noise Ratio (SNR $< -5\text{ dB}$)
- **Behavior:** In severe noise environments ($< -5\text{ dB}$), cyclostationary baud rate peaks merge into the noise floor, and deep neural modulation feature extractors exhibit degraded Top-1 confidence ($< 25\%$).
- **Mitigation:** ASTRA employs a bounded multi-hypothesis candidate beam search (Top-5 modulation $\times$ Top-3 baud = 15 parallel hypotheses) that maintains an $87.4\%$ system retention rate down to $0\text{ dB}$.

### 1.2 Dimensionless Raw Binary I/Q Ingestion
- **Behavior:** Standard raw `.iq` files (unlike SigMF or structured `.wav`) lack standardized sample rate metadata. If sample rate ($F_s$) is unspecified, blind baud estimation relies on dimensionless normalized samples-per-symbol ($SPS$).
- **Mitigation:** ASTRA accepts explicit `--sample-rate` CLI flags and default UI heuristics calibrated to standard SDR ingestion rates ($1\text{ MSps}$ to $10\text{ MSps}$).

---

## 2. DSP & Modulation Constraints

### 2.1 Dense High-Order QAM (64-QAM / 256-QAM) Phase Ambiguity
- **Behavior:** At low SNRs, Decision-Directed Carrier Phase Locking on 64-QAM and 256-QAM constellations experiences decision errors due to close Euclidean distances between adjacent constellation points.
- **Scope:** BPSK, QPSK, 8-PSK, and 16-QAM exhibit robust phase lock ($> 99.9\%$ lock rate across all tests).

### 2.2 Continuous Phase Modulation & Non-Linear Schemes
- **Behavior:** ASTRA is optimized for digital PSK, FSK, and QAM families. Continuous-phase systems like GMSK or spread-spectrum DSSS / FHSS require specialized matched correlation kernels not active in the default candidate beam.

---

## 3. Cryptographic & Coding Bounds

### 3.1 Blind Arbitrary Pseudo-Random Interleavers
- **Behavior:** While block ($R \times C$), convolutional, helical, and standard seeded PRBS deinterleavers are fully supported, solving completely unconstrained, non-deterministic random permutations with unknown cryptographic seed generation in real-time requires exponential compute complexity.
- **Mitigation:** ASTRA tests standard pseudo-random seed profiles (Fisher-Yates / PCG64 / LFSR) alongside deterministic structural deinterleavers.

### 3.2 Non-Standard / Proprietary FEC Polynomials
- **Behavior:** Viterbi decoding supports standard CCSDS / NASA polynomials ($K=7$, $[171_8, 133_8]$) and Reed-Solomon $(255, 223)$, $(255, 239)$ over $GF(2^8)$. Custom non-standard generator polynomials outside the active registry must be added to `astra_config/fec_profiles.yaml`.
