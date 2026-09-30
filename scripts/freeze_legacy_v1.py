"""
Freeze Legacy Modulation V1 pipeline.
Copies legacy checkpoints, configs, preprocessing, metrics, class mappings,
fusion weights, and benchmark outputs into legacy_modulation_v1/.
"""

import os
import shutil
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = ROOT / "legacy_modulation_v1"

(LEGACY_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(LEGACY_DIR / "configs").mkdir(parents=True, exist_ok=True)
(LEGACY_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(LEGACY_DIR / "preprocessing").mkdir(parents=True, exist_ok=True)

# 1. Checkpoints
checkpoints_to_freeze = [
    ROOT / "checkpoints" / "astra_resnet1d_modulation_v2.pt",
    ROOT / "checkpoints" / "astra_spectrogram_cnn_v2.pt",
    ROOT / "outputs" / "CSPB.ML.2018R2" / "best_model.pt",
    ROOT / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "checkpoints" / "best_model_cspb_baseline.pt",
]

for src in checkpoints_to_freeze:
    if src.exists():
        dst = LEGACY_DIR / "checkpoints" / src.name
        shutil.copy2(src, dst)
        print(f"[FROZEN CKPT] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

# 2. Configs
configs_to_freeze = [
    (ROOT / "astra_fusion" / "configs" / "fusion_config.yaml", "fusion_config.yaml"),
    (ROOT / "astra_modulation_2d" / "configs" / "cspb.yaml", "spectrogram_cspb.yaml"),
    (ROOT / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "config.yaml", "starter_config.yaml"),
]

for src, dst_name in configs_to_freeze:
    if src.exists():
        dst = LEGACY_DIR / "configs" / dst_name
        shutil.copy2(src, dst)
        print(f"[FROZEN CFG] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

# 3. Metrics
metrics_to_freeze = [
    ROOT / "checkpoints" / "final_benchmark_results.json",
    ROOT / "checkpoints" / "v2_evaluation_report.json",
    ROOT / "checkpoints" / "v2_fusion_evaluation.json",
    ROOT / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "eval_results" / "cspb_baseline_report.json",
]

for src in metrics_to_freeze:
    if src.exists():
        dst = LEGACY_DIR / "metrics" / src.name
        shutil.copy2(src, dst)
        print(f"[FROZEN METRIC] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

# 4. Preprocessing and Models
code_to_freeze = [
    (ROOT / "astra_fusion" / "src" / "adapters.py", "fusion_adapters.py"),
    (ROOT / "astra_modulation_2d" / "src" / "preprocessing.py", "spectrogram_preprocessing.py"),
    (ROOT / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "src" / "cspb_loader.py", "cspb_loader.py"),
    (ROOT / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "src" / "model.py", "model_resnet1d_v1.py"),
    (ROOT / "astra_modulation_2d" / "src" / "model.py", "model_spectrogram2d_v1.py"),
]

for src, dst_name in code_to_freeze:
    if src.exists():
        dst = LEGACY_DIR / "preprocessing" / dst_name
        shutil.copy2(src, dst)
        print(f"[FROZEN CODE] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

manifest = {
    "version": "legacy_modulation_v1",
    "description": "Frozen snapshot of ASTRA modulation classification V1 architecture, weights, and metrics.",
    "models": {
        "resnet1d": {
            "checkpoint": "checkpoints/astra_resnet1d_modulation_v2.pt",
            "classes": ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "DQPSK", "MSK", "16QAM", "64QAM", "256QAM"]
        },
        "spectrogram_cnn2d": {
            "checkpoint": "checkpoints/astra_spectrogram_cnn_v2.pt",
            "classes": ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "DQPSK", "MSK", "16QAM", "64QAM", "256QAM"]
        },
        "legacy_cspb_cnn2d": {
            "checkpoint": "outputs/CSPB.ML.2018R2/best_model.pt",
            "classes": ["bpsk", "qpsk", "8psk", "dqpsk", "msk", "16qam", "64qam", "256qam"]
        }
    },
    "fusion": {
        "mode": "weighted_probability",
        "weights": {"resnet1d": 0.75, "spectrogram2d": 0.25}
    },
    "baseline_untouched_benchmark": "metrics/final_benchmark_results.json"
}

with open(LEGACY_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("V1 freeze completed successfully.")
