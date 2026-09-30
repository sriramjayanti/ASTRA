# ASTRA Model 1 Starter Pack

Build the first ASTRA ML component: a 1D ResNet CNN for modulation classification from raw IQ samples.

Initial classes:
- bpsk
- qpsk
- 8psk
- dqpsk
- msk
- 16qam
- 64qam
- 256qam

Dataset: CSPB.ML.2018R2

Expected input tensor: [batch, 2, 2048]
- channel 0 = I
- channel 1 = Q

Expected output: full modulation probability vector + Top-K candidates.

Deliverables:
1. Dataset parser
2. File-level train/val/test split
3. 1D ResNet model
4. Training script
5. Evaluation script
6. Accuracy, precision, recall, Macro-F1
7. Confusion matrix
8. Accuracy vs SNR
9. Saved model weights
10. Inference function returning Top-K probabilities

Read MODEL_1_PROMPT.md first.
