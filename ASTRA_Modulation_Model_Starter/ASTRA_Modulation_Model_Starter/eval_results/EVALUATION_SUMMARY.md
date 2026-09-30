# ASTRA Model 1 Evaluation Report

## Summary Metrics
- **Total Test Windows**: 60
- **Overall Accuracy**: **38.33%**
- **Macro Precision**: 0.5095
- **Macro Recall**: 0.4219
- **Macro-F1 Score**: **0.3679**
- **Weighted-F1 Score**: 0.3258

## Per-Class Metrics
| Modulation | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| `bpsk` | 80.00% | 100.00% | 0.8889 | 8 |
| `qpsk` | 100.00% | 12.50% | 0.2222 | 8 |
| `8psk` | 0.00% | 0.00% | 0.0000 | 8 |
| `dqpsk` | 0.00% | 0.00% | 0.0000 | 8 |
| `msk` | 100.00% | 100.00% | 1.0000 | 4 |
| `16qam` | 100.00% | 25.00% | 0.4000 | 8 |
| `64qam` | 27.59% | 100.00% | 0.4324 | 8 |
| `256qam` | 0.00% | 0.00% | 0.0000 | 8 |

## Accuracy vs. SNR (dB)
| SNR (dB) | Accuracy (%) | Windows |
|----------|--------------|---------|
| -5.0 dB | 50.00% | 16 |
| +0.0 dB | 25.00% | 8 |
| +5.0 dB | 12.50% | 8 |
| +10.0 dB | 100.00% | 4 |
| +15.0 dB | 50.00% | 16 |
| +20.0 dB | 0.00% | 8 |

## Generated Visualizations
- Confusion Matrix: `eval_results/confusion_matrix.png`
- Accuracy vs. SNR Curve: `eval_results/accuracy_vs_snr.png`
