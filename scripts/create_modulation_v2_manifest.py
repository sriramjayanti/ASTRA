"""
Generate canonical modulation_v2_manifest.json.
Documents SHA256, architectures, classes, preprocessing, dataset version, and metrics.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import MODULATION_CLASSES_V2, CLASS_SCHEMA_VERSION, NUM_CLASSES_V2

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

ckpt_1d = ROOT / "checkpoints" / "astra_resnet1d_v2.pt"
ckpt_2d = ROOT / "checkpoints" / "astra_spectrogram_cnn_v2.pt"

with open(ROOT / "checkpoints" / "resnet1d_v2_metrics.json", "r") as f:
    m1d = json.load(f)

with open(ROOT / "checkpoints" / "spectrogram_cnn2d_v2_metrics.json", "r") as f:
    m2d = json.load(f)

with open(ROOT / "checkpoints" / "v2_fusion_metrics.json", "r") as f:
    mfusion = json.load(f)

with open(ROOT / "checkpoints" / "blind_benchmark_v2_results.json", "r") as f:
    mblind = json.load(f)

manifest = {
    "manifest_version": "2.0.0",
    "class_schema_version": CLASS_SCHEMA_VERSION,
    "classes": MODULATION_CLASSES_V2,
    "num_classes": NUM_CLASSES_V2,
    "dataset_version": "ASTRA_MODULATION_DATASET_V2",
    "models": {
        "astra_resnet1d_modulation_v2": {
            "model_id": "astra_resnet1d_modulation_v2",
            "architecture": "ResNet1DV2",
            "file_path": str(ckpt_1d.relative_to(ROOT)),
            "sha256": sha256_file(ckpt_1d),
            "input_shape": [2, 2048],
            "preprocessing": "IQPreprocessorV2 (DC removal + RMS normalization)",
            "metrics": {
                "test_top1": m1d["test"]["top1_accuracy"],
                "test_top3": m1d["test"]["top3_accuracy"],
                "test_macro_f1": m1d["test"]["macro_f1"],
                "ood_top1": m1d["ood"]["top1_accuracy"],
            }
        },
        "astra_spectrogram_cnn_modulation_v2": {
            "model_id": "astra_spectrogram_cnn_modulation_v2",
            "architecture": "SpectrogramCNN2DV2",
            "file_path": str(ckpt_2d.relative_to(ROOT)),
            "sha256": sha256_file(ckpt_2d),
            "input_shape": [1, 128, 128],
            "preprocessing": "Centered Log-Power STFT [128x128] + Standardized",
            "metrics": {
                "test_top1": m2d["test"]["top1_accuracy"],
                "test_top3": m2d["test"]["top3_accuracy"],
                "test_macro_f1": m2d["test"]["macro_f1"],
                "ood_top1": m2d["ood"]["top1_accuracy"],
            }
        },
        "astra_fusion_modulation_v2": {
            "model_id": "astra_fusion_modulation_v2",
            "mode": "calibrated_weighted_probability",
            "weights": mfusion["optimal_weights"],
            "rf_support_beta": mfusion["rf_support_beta"],
            "test_top1": mfusion["test_metrics"]["top1_accuracy"],
            "test_top3": mfusion["test_metrics"]["top3_accuracy"],
            "test_macro_f1": mfusion["test_metrics"]["macro_f1"],
            "blind_benchmark_top1": mblind["metrics"]["Modulation Top-1"],
            "blind_benchmark_top3": mblind["metrics"]["Modulation Top-3"],
            "blind_benchmark_macro_f1": mblind["metrics"]["Macro-F1"],
            "blind_benchmark_noise_fpr": mblind["metrics"]["Noise false-positive rate"],
        }
    }
}

out_path = ROOT / "modulation_v2_manifest.json"
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Created canonical model manifest at {out_path}")
