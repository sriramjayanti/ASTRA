# ASTRA 2D Spectrogram CNN Modulation Classifier

A production-ready Deep 2D Residual Convolutional Neural Network with Squeeze-and-Excitation (SE) Channel Attention for time-frequency modulation classification in **ASTRA (Automated Signal Analysis & Recovery Assistant)**.

---

## 1. Role in ASTRA Multi-Branch Architecture

The 2D Spectrogram Classifier acts as a supporting time-frequency evidence branch alongside the 1D raw-IQ classifier:

```
                      Raw Complex Baseband IQ [N]
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│  1D ResNet IQ    │                            │  STFT Generator  │
│  Raw Waveform    │                            │  (Centered PSD)  │
│    Classifier    │                            └─────────┬────────┘
└─────────┬────────┘                                      │
          │                                               ▼
          │                                     ┌──────────────────┐
          │                                     │  2D ResNet + SE  │
          │                                     │ Spectrogram CNN  │
          │                                     └─────────┬────────┘
          │                                               │
          │ (1D Logits / Probs)                           │ (2D Logits / Probs)
          └───────────────────────┬───────────────────────┘
                                  ▼
                     ┌───────────────────────────┐
                     │   ASTRA Fusion Engine     │
                     │  (Score & Feature Fusion) │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ Top-K Candidate Generator │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ Receiver Demod & Recovery │
                     └───────────────────────────┘
```

---

## 2. STFT & Time-Frequency Spectral Representation

1. **Input Signal:**
   $$x[n] = I[n] + j Q[n], \quad n \in [0, N-1], \quad N = 2048$$
2. **Centralized IQ Preprocessing:**
   - Finite validation & NaN/Inf replacement
   - Zero-mean DC removal: $x[n] \leftarrow x[n] - \mu_x$
   - Unit RMS amplitude normalization: $x[n] \leftarrow \frac{x[n]}{\sqrt{\frac{1}{N}\sum |x[n]|^2} + \epsilon}$
3. **Complex Baseband STFT:**
   $$\text{STFT}(f, t) = \sum_{m} x[m] \cdot w[m - t \cdot H] \cdot e^{-j 2 \pi f m}$$
   - Window: Hann ($L = 128$)
   - FFT size: $N_{\text{fft}} = 128$
   - Hop length: $H = 32$
   - Two-sided frequency spectrum (`onesided=False`)
   - Centered DC frequency shift: `torch.fft.fftshift`
4. **Log-Power Spectrogram:**
   $$S(f, t) = 10 \log_{10}(|\text{STFT}(f, t)|^2 + 10^{-8})$$
5. **Standardization:**
   $$S_{\text{norm}}(f, t) = \frac{S(f, t) - \mu_S}{\sigma_S + 10^{-6}}$$
   - Output Tensor Shape: $[B, 1, F=128, T=65]$ (numerical float32 tensor without lossy PNG image conversion)

---

## 3. 2D CNN Architecture with SE Attention

- **Stem:** $\text{Conv2D}(1 \to 32, 3 \times 3) \to \text{BatchNorm2D} \to \text{ReLU}$
- **Stage 1:** 2 Residual Blocks ($32 \to 64$, first block stride 2)
- **Stage 2:** 2 Residual Blocks ($64 \to 128$, first block stride 2)
- **Stage 3:** 2 Residual Blocks ($128 \to 256$, first block stride 2)
- **Squeeze-and-Excitation Attention:** Adaptive channel recalibration with reduction ratio $r = 16$:
  $$\mathbf{z} = \text{GlobalAvgPool}(\mathbf{X}) \in \mathbb{R}^{256}$$
  $$\mathbf{s} = \sigma(\mathbf{W}_2 \, \text{ReLU}(\mathbf{W}_1 \mathbf{z}))$$
  $$\mathbf{\tilde{X}} = \mathbf{X} \odot \mathbf{s}$$
- **Global Average Pooling:** $\text{AdaptiveAvgPool2D}(1, 1) \to 256\text{-D feature embedding}$
- **Classifier Head:** $\text{Linear}(256 \to 128) \to \text{BatchNorm1D} \to \text{ReLU} \to \text{Dropout}(0.25) \to \text{Linear}(128 \to \text{num\_classes})$
- **Output:** Raw logits $[B, \text{num\_classes}]$. Softmax is applied strictly during inference.

---

## 4. Supported Dataset Modes

- **Mode A (CSPB.ML.2018R2):**
  - 8 Classes: `bpsk`, `qpsk`, `8psk`, `dqpsk`, `msk`, `16qam`, `64qam`, `256qam`
  - Config: `configs/cspb.yaml`
- **Mode B (ASTRA Synthetic 10-Class):**
  - 10 Classes: `2fsk`, `4fsk`, `bpsk`, `qpsk`, `8psk`, `dqpsk`, `msk`, `16qam`, `64qam`, `256qam`
  - Config: `configs/astra_synthetic.yaml`

---

## 5. Usage Commands

### Run Unit Tests
```bash
pytest astra_modulation_2d/tests -v
```

### Run Tiny-Set Overfit Sanity Check (32 samples)
```bash
python astra_modulation_2d/src/train.py --config astra_modulation_2d/configs/cspb.yaml --tiny_overfit
```

### Full Training
```bash
python astra_modulation_2d/src/train.py --config astra_modulation_2d/configs/cspb.yaml
```

### Evaluation
```bash
python astra_modulation_2d/src/evaluate.py --config astra_modulation_2d/configs/cspb.yaml --checkpoint outputs/CSPB.ML.2018R2/best_model.pt
```
