# EXACT TASK PROMPT — ASTRA MODEL 1

You are implementing the first ML/DL model for ASTRA — Automated Signal Analysis & Recovery Assistant.

## Goal
Build a reliable 1D ResNet CNN modulation classifier that receives a fixed-length raw IQ window and outputs modulation probabilities.

Do not work on GUI, FEC, interleaving, bitstream Transformer, or final integration yet.

## Dataset
Use CSPB.ML.2018R2 only for the first version.

Initial classes:
bpsk, qpsk, 8psk, dqpsk, msk, 16qam, 64qam, 256qam

Truth metadata fields:
1. signal index
2. modulation type
3. base symbol period T0
4. carrier offset
5. excess bandwidth / RRC roll-off
6. upsample factor U
7. downsample factor D
8. in-band SNR dB
9. noise spectral density dB

Normalized symbol rate:
f_sym = (1/T0) * (D/U)

Primary target for this model: modulation type.
Retain SNR, CFO, roll-off and symbol-rate-related values for analysis.

## Critical split rule
Split by COMPLETE SIGNAL FILE before windowing:
- 70% train
- 15% validation
- 15% test

No window from the same signal file may appear in more than one split.
Use a fixed seed and save split manifests.

## .tim reader
The CSPB signal files contain interleaved real and imaginary samples.

Before hard-coding the reader:
- verify the exact binary dtype from the official CSPB read_binary.m or dataset documentation;
- do not guess the dtype.

Return:
complex_samples: np.ndarray[complex64]

Then convert each window to:
[2, N]
where channel 0 = I and channel 1 = Q.

## Windowing
Start with:
N = 2048 samples

Make window size configurable so later we can test:
1024, 2048, 4096, 8192.

No window may cross file boundaries.

## Preprocessing
Required:
1. Parse complex IQ correctly
2. Remove NaN/Inf
3. Optional DC removal
4. RMS amplitude normalization
5. Split complex IQ into I and Q channels

Recommended:
iq = iq - mean(iq)
iq = iq / (sqrt(mean(abs(iq)^2)) + eps)

Do not aggressively filter or resample the training data in the baseline.

## Model
Framework: PyTorch

Architecture:
Input [B,2,N]
→ Conv1D
→ BatchNorm1D
→ ReLU
→ Residual Blocks
→ Adaptive Global Average Pooling
→ Fully Connected Layer
→ logits

Use Softmax only for inference.
Use CrossEntropyLoss directly on logits during training.

Why:
- IQ is 1D sequential data
- Conv1D learns local waveform/phase/amplitude patterns
- residual connections allow deeper stable networks
- global pooling reduces parameters
- full probabilities are needed by ASTRA Candidate Engine

## Training
Loss: CrossEntropyLoss
Optimizer: AdamW
Initial LR: 1e-3
Batch size: 64
Epochs: 30–50
Scheduler: ReduceLROnPlateau or cosine annealing
Early stopping: validation Macro-F1 or validation loss

Save:
- best checkpoint
- final checkpoint
- config
- class mapping
- random seed
- split manifests

## Evaluation
Report:
- accuracy
- precision
- recall
- Macro-F1
- weighted F1
- confusion matrix
- per-class metrics
- accuracy vs SNR

Do NOT report only one accuracy value.

## Inference API
Create:

predict_modulation(iq_window)

Output format:
{
  "predicted_class": "qpsk",
  "confidence": 0.91,
  "top_k": [
    {"class": "qpsk", "probability": 0.91},
    {"class": "8psk", "probability": 0.05},
    {"class": "dqpsk", "probability": 0.02}
  ],
  "probabilities": {
    "bpsk": ...,
    "qpsk": ...,
    "8psk": ...,
    "dqpsk": ...,
    "msk": ...,
    "16qam": ...,
    "64qam": ...,
    "256qam": ...
  }
}

ASTRA needs full probabilities and Top-K, not only argmax.

## First milestone
Before full training:
1. Read 10–20 .tim files correctly
2. Match each file to truth metadata
3. Print first IQ samples
4. Plot I, Q, magnitude and quick PSD
5. Create [2,2048] windows
6. Run one batch through network
7. Confirm output shape [batch,8]
8. Intentionally overfit a tiny subset to prove the pipeline works
9. Only then run full training

## Definition of done
Complete when:
- .tim loading is correct
- metadata mapping is correct
- zero train/val/test leakage
- 1D ResNet trains
- metrics are reported
- confusion matrix generated
- accuracy-vs-SNR generated
- model saved
- inference returns Top-K probabilities
- run is reproducible

At the end provide:
1. source code
2. trained model
3. metrics report
4. confusion matrix
5. accuracy-vs-SNR chart
6. split manifests
7. training command
8. evaluation command
9. unresolved issues
