# ASTRA: Future Roadmap & Engineering Horizons

This document outlines planned enhancements, research initiatives, and future architectural expansions for the ASTRA autonomous signal intelligence platform.

---

## 1. Hardware & Real-Time SDR Deployment

- **Native UHD / SoapySDR Driver Integration:** Direct live streaming ingestion from USRP, RTL-SDR, HackRF, and BladeRF hardware over high-speed USB 3.0 / 10 GbE interfaces.
- **CUDA / C++ Accelerated DSP Kernels:** Re-implementing critical timing recovery loops (Gardner TED), matched polyphase filtering, and STFT compute passes in custom CUDA C++ kernels to enable sustained $> 50\text{ MSps}$ continuous streaming.
- **FPGA Offloading:** Synthesizing coarse CFO wiping and decimation front-ends for Xilinx / AMD Zynq Ultrascale+ RFSoC edge devices.

---

## 2. Advanced Deep Learning & Modulation Intelligence

- **Self-Supervised RF Foundation Models:** Training masked spectrogram autoencoders and contrastive representation models (e.g., RF-BERT / SignalTransformer) across massive unlabelled multi-gigabyte RF spectrum databases.
- **Low-SNR Dense-QAM Classification:** Incorporating higher-order cumulant features ($C_{40}, C_{42}, C_{60}$) and constellation density clustering into the deep feature fusion layer to boost 64-QAM / 256-QAM Top-1 classification at $< 5\text{ dB}$ SNR.
- **Dynamic Candidate Beam Auto-Tuning:** Reinforcement-learning-guided early stopping in the candidate hypothesis engine to dynamically expand beam width for ambiguous signals while terminating early on clean captures.

---

## 3. Protocol & Framing Expansion

- **Automated Protocol Dissector Generator:** Neural reverse-engineering of unknown packet structures to generate Wireshark / Scapy dissector definitions on the fly.
- **Spread-Spectrum (DSSS / FHSS) Despreading:** Blind estimation of hopping sequences and pseudo-random spreading codes via cyclic bispectrum analysis.
- **Generalized Non-Binary LDPC & Polar Decoders:** Extending Stage 9 FEC testing to encompass 5G NR Polar codes and non-binary LDPC profiles.

---

## 4. Cloud & Collaborative Intelligence

- **Distributed Model Repository & Checkpoint Sync:** Lightweight cloud sync for automated weight updates and newly characterized modulation profiles.
- **Multi-Sensor TDOA / FDOA Triangulation:** Ingesting multi-node IQ captures for simultaneous blind decoding and emitter geolocation.
