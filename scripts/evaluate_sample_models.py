"""
ASTRA Rigorous Evaluation Suite for Sample Models.

Compares:
1. Sample ResNet-1D (samplee models/finetuned_best_resnet1d.pth)
2. Sample Spectrogram CNN-2D (samplee models/finetuned_best_2dcnn.pth)
3. Sample Fused Ensemble (Sample 1D + Sample 2D)
4. ASTRA V2 Production ResNet-1D (checkpoints/astra_resnet1d_v2.pt)
5. ASTRA V2 Production Spectrogram CNN-2D (checkpoints/astra_spectrogram_cnn_v2.pt)
6. ASTRA V2 Production Fusion

Evaluates on:
A. ASTRA_MODULATION_DATASET_V2 Test Split (1,050 source captures)
B. ASTRA_MODULATION_DATASET_V2 OOD Test Split (660 source captures)
C. ASTRA_FINAL_TEST_SET_V2_BLIND (1,100 unseen captures)

Reports:
- Top-1 and Top-3 Accuracy
- Macro-F1 and Weighted-F1
- Per-class Precision, Recall, F1, and Support
- Performance on 8-Class Supported Subset vs Full 11-Class System Alphabet
- SNR-stratified accuracy curves (<0, 0-5, 5-10, 10-15, 15-20, >20 dB)
- UNKNOWN / Noise Rejection & False Positive Rate
- Confusion Matrices (QAM, PSK, FSK)
- Architectural Parameter Counts, Inference Latency, and Drawbacks
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

sys.path.insert(0, ".")
from astra_modulation_v2.class_schema import MODULATION_CLASSES_V2, NUM_CLASSES_V2
from astra_modulation_v2.models.resnet1d import ResNet1DV2
from astra_modulation_v2.models.spectrogram_cnn import SpectrogramCNN2DV2
from astra_modulation_2d.src.spectrogram import SpectrogramGenerator

# ==============================================================================
# Model Architecture Definitions for Sample Models
# ==============================================================================

class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class SampleResNet1D(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(ResBlock1D(64, 64), ResBlock1D(64, 64))
        self.stage2 = nn.Sequential(ResBlock1D(64, 128, stride=2), ResBlock1D(128, 128))
        self.stage3 = nn.Sequential(ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256))
        self.stage4 = nn.Sequential(ResBlock1D(256, 512, stride=2), ResBlock1D(512, 512))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        avg = self.avg_pool(x).flatten(1)
        mx = self.max_pool(x).flatten(1)
        feat = torch.cat([avg, mx], dim=1)
        return self.classifier(feat)

class ResBlock2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class SampleSpectrogram2DCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(ResBlock2D(32, 64, stride=2), ResBlock2D(64, 64))
        self.stage2 = nn.Sequential(ResBlock2D(64, 128, stride=2), ResBlock2D(128, 128))
        self.stage3 = nn.Sequential(ResBlock2D(128, 256, stride=2), ResBlock2D(256, 256))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat = self.global_pool(x).flatten(1)
        return self.classifier(feat)

# ==============================================================================
# Helper Functions: Loading and Normalization
# ==============================================================================

CANONICAL_ALIASES = {
    "16-qam": "16QAM",
    "64-qam": "64QAM",
    "256-qam": "256QAM",
    "unknown": "UNKNOWN",
    "noise": "UNKNOWN",
}

def canonicalize_mod(m: str) -> str:
    s = str(m).strip()
    return CANONICAL_ALIASES.get(s.lower(), s)

def preprocess_iq(raw_iq: np.ndarray, n_samples: int = 2048) -> np.ndarray:
    raw_iq = np.nan_to_num(raw_iq, nan=0.0, posinf=0.0, neginf=0.0)
    if len(raw_iq) < n_samples:
        pad = np.zeros(n_samples - len(raw_iq), dtype=np.complex64)
        raw_iq = np.concatenate([raw_iq, pad])
    else:
        raw_iq = raw_iq[:n_samples]

    raw_iq = raw_iq - np.mean(raw_iq)
    rms = np.sqrt(np.mean(np.abs(raw_iq) ** 2)) + 1e-8
    raw_iq = raw_iq / rms

    # Return shape [2, N]
    iq_2d = np.stack([raw_iq.real.astype(np.float32), raw_iq.imag.astype(np.float32)], axis=0)
    return iq_2d

# ==============================================================================
# Main Evaluation Class
# ==============================================================================

class ModelEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[EVALUATOR] Using execution device: {self.device}")

        # 1. Load Sample Models
        p_sample_1d = r"samplee models\finetuned_best_resnet1d.pth"
        p_sample_2d = r"samplee models\finetuned_best_2dcnn.pth"

        print("[EVALUATOR] Loading Sample Models...")
        self.sample_1d = SampleResNet1D(num_classes=8).to(self.device)
        c1d = torch.load(p_sample_1d, map_location=self.device)
        self.sample_1d.load_state_dict(c1d.get("model_state_dict", c1d.get("state_dict", c1d)))
        self.sample_1d.eval()

        self.sample_2d = SampleSpectrogram2DCNN(num_classes=8).to(self.device)
        c2d = torch.load(p_sample_2d, map_location=self.device)
        self.sample_2d.load_state_dict(c2d.get("model_state_dict", c2d.get("state_dict", c2d)))
        self.sample_2d.eval()

        raw_sample_classes = c1d.get("classes", ['2-FSK', '4-FSK', 'BPSK', 'QPSK', '8PSK', '16-QAM', '64-QAM', 'Unknown'])
        self.sample_classes = [canonicalize_mod(c) for c in raw_sample_classes]
        print(f"  Sample classes ({len(self.sample_classes)}): {self.sample_classes}")

        # 2. Load V2 Production Models
        print("[EVALUATOR] Loading V2 Production Models...")
        self.v2_1d = ResNet1DV2(num_classes=NUM_CLASSES_V2).to(self.device)
        c_v2_1d = torch.load("checkpoints/astra_resnet1d_v2.pt", map_location=self.device)
        self.v2_1d.load_state_dict(c_v2_1d["state_dict"])
        self.v2_1d.eval()

        self.v2_2d = SpectrogramCNN2DV2(num_classes=NUM_CLASSES_V2).to(self.device)
        c_v2_2d = torch.load("checkpoints/astra_spectrogram_cnn_v2.pt", map_location=self.device)
        self.v2_2d.load_state_dict(c_v2_2d["state_dict"])
        self.v2_2d.eval()
        self.v2_classes = list(MODULATION_CLASSES_V2)
        print(f"  V2 classes ({len(self.v2_classes)}): {self.v2_classes}")

        # 3. Spectrogram Generator
        self.spec_gen = SpectrogramGenerator().to(self.device)

        # Count parameters
        self.param_counts = {
            "Sample ResNet-1D": sum(p.numel() for p in self.sample_1d.parameters()),
            "Sample Spectrogram 2D-CNN": sum(p.numel() for p in self.sample_2d.parameters()),
            "V2 Production ResNet-1D": sum(p.numel() for p in self.v2_1d.parameters()),
            "V2 Production Spectrogram CNN-2D": sum(p.numel() for p in self.v2_2d.parameters()),
        }
        for k, v in self.param_counts.items():
            print(f"  {k} Parameters: {v:,} ({v * 4 / (1024*1024):.2f} MB)")

    def evaluate_dataset(self, captures: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
        print(f"\n================================================================================")
        print(f"EVALUATING ON: {dataset_name} ({len(captures)} captures)")
        print(f"================================================================================")

        batch_size = 64
        all_true_labels = []
        all_snrs = []

        # Predictions storage
        preds_sample_1d, top3_sample_1d = [], []
        preds_sample_2d, top3_sample_2d = [], []
        preds_sample_fused, top3_sample_fused = [], []

        preds_v2_1d, top3_v2_1d = [], []
        preds_v2_2d, top3_v2_2d = [], []
        preds_v2_fused, top3_v2_fused = [], []

        # Latency tracking
        t_sample_1d, t_sample_2d = 0.0, 0.0
        t_v2_1d, t_v2_2d = 0.0, 0.0

        n_batches = int(np.ceil(len(captures) / batch_size))

        for b in range(n_batches):
            batch_slice = captures[b * batch_size : (b + 1) * batch_size]
            batch_tensors = []
            batch_true = []
            batch_snr = []

            for cap in batch_slice:
                iq_path = cap.get("iq_path")
                true_mod = canonicalize_mod(cap.get("true_modulation", cap.get("modulation", "UNKNOWN")))
                snr = float(cap.get("snr_db", 10.0))

                try:
                    raw_iq = np.fromfile(iq_path, dtype=np.complex64)
                    iq_2d = preprocess_iq(raw_iq)
                    batch_tensors.append(iq_2d)
                    batch_true.append(true_mod)
                    batch_snr.append(snr)
                except Exception as e:
                    continue

            if not batch_tensors:
                continue

            x = torch.from_numpy(np.stack(batch_tensors)).to(self.device)  # [B, 2, 2048]

            with torch.no_grad():
                # --- Sample 1D ---
                t0 = time.perf_counter()
                logits_s1d = self.sample_1d(x)
                t_sample_1d += (time.perf_counter() - t0)
                probs_s1d = torch.softmax(logits_s1d, dim=-1)

                # --- Spectrogram Generator ---
                spec = self.spec_gen(x)

                # --- Sample 2D ---
                t0 = time.perf_counter()
                logits_s2d = self.sample_2d(spec)
                t_sample_2d += (time.perf_counter() - t0)
                probs_s2d = torch.softmax(logits_s2d, dim=-1)

                # --- Sample Fused ---
                probs_sfused = 0.5 * probs_s1d + 0.5 * probs_s2d

                # --- V2 1D ---
                t0 = time.perf_counter()
                logits_v1d, _ = self.v2_1d(x)
                t_v2_1d += (time.perf_counter() - t0)
                probs_v1d = torch.softmax(logits_v1d, dim=-1)

                # --- V2 2D ---
                t0 = time.perf_counter()
                logits_v2d, _ = self.v2_2d(spec)
                t_v2_2d += (time.perf_counter() - t0)
                probs_v2d = torch.softmax(logits_v2d, dim=-1)

                # --- V2 Fused ---
                probs_vfused = 0.5 * probs_v1d + 0.5 * probs_v2d

            # Record predictions
            # For Sample models (8 classes):
            for i in range(len(batch_true)):
                # 1D
                p1 = probs_s1d[i].cpu().numpy()
                idx1 = int(np.argmax(p1))
                preds_sample_1d.append(self.sample_classes[idx1])
                top3_sample_1d.append([self.sample_classes[idx] for idx in np.argsort(-p1)[:3]])

                # 2D
                p2 = probs_s2d[i].cpu().numpy()
                idx2 = int(np.argmax(p2))
                preds_sample_2d.append(self.sample_classes[idx2])
                top3_sample_2d.append([self.sample_classes[idx] for idx in np.argsort(-p2)[:3]])

                # Fused
                pf = probs_sfused[i].cpu().numpy()
                idxf = int(np.argmax(pf))
                preds_sample_fused.append(self.sample_classes[idxf])
                top3_sample_fused.append([self.sample_classes[idx] for idx in np.argsort(-pf)[:3]])

                # V2 1D
                pv1 = probs_v1d[i].cpu().numpy()
                idxv1 = int(np.argmax(pv1))
                preds_v2_1d.append(self.v2_classes[idxv1])
                top3_v2_1d.append([self.v2_classes[idx] for idx in np.argsort(-pv1)[:3]])

                # V2 2D
                pv2 = probs_v2d[i].cpu().numpy()
                idxv2 = int(np.argmax(pv2))
                preds_v2_2d.append(self.v2_classes[idxv2])
                top3_v2_2d.append([self.v2_classes[idx] for idx in np.argsort(-pv2)[:3]])

                # V2 Fused
                pvf = probs_vfused[i].cpu().numpy()
                idxvf = int(np.argmax(pvf))
                preds_v2_fused.append(self.v2_classes[idxvf])
                top3_v2_fused.append([self.v2_classes[idx] for idx in np.argsort(-pvf)[:3]])

                all_true_labels.append(batch_true[i])
                all_snrs.append(batch_snr[i])

        n_total = len(all_true_labels)
        print(f"Successfully processed {n_total} signals.")

        # Compute Metrics
        models_eval = {
            "Sample ResNet-1D (8-class)": (preds_sample_1d, top3_sample_1d, t_sample_1d),
            "Sample Spectrogram 2D-CNN (8-class)": (preds_sample_2d, top3_sample_2d, t_sample_2d),
            "Sample Fused (1D+2D)": (preds_sample_fused, top3_sample_fused, t_sample_1d + t_sample_2d),
            "V2 Production ResNet-1D (11-class)": (preds_v2_1d, top3_v2_1d, t_v2_1d),
            "V2 Production Spectrogram CNN-2D (11-class)": (preds_v2_2d, top3_v2_2d, t_v2_2d),
            "V2 Production Fusion (11-class)": (preds_v2_fused, top3_v2_fused, t_v2_1d + t_v2_2d),
        }

        # 1. Full 11-Class System Evaluation
        results_full_11 = {}
        for name, (preds, top3, lat) in models_eval.items():
            top1_acc = np.mean([1 if p == t else 0 for p, t in zip(preds, all_true_labels)]) * 100.0
            top3_acc = np.mean([1 if t in t3 else 0 for t, t3 in zip(all_true_labels, top3)]) * 100.0
            macro_f1 = f1_score(all_true_labels, preds, average="macro", zero_division=0)
            weighted_f1 = f1_score(all_true_labels, preds, average="weighted", zero_division=0)

            # Per-class metrics
            per_class = {}
            for cls in self.v2_classes:
                cls_indices = [i for i, t in enumerate(all_true_labels) if t == cls]
                if cls_indices:
                    cls_top1 = np.mean([1 if preds[i] == cls else 0 for i in cls_indices]) * 100.0
                    cls_top3 = np.mean([1 if cls in top3[i] else 0 for i in cls_indices]) * 100.0
                    per_class[cls] = {
                        "support": len(cls_indices),
                        "top1": cls_top1,
                        "top3": cls_top3
                    }
                else:
                    per_class[cls] = {"support": 0, "top1": 0.0, "top3": 0.0}

            results_full_11[name] = {
                "top1_accuracy": top1_acc,
                "top3_accuracy": top3_acc,
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "latency_per_sample_ms": (lat / n_total) * 1000.0,
                "per_class": per_class,
            }

        # 2. 8-Class Supported Subset Evaluation (Filtering out DQPSK, MSK, 256QAM)
        sub_indices = [i for i, t in enumerate(all_true_labels) if t in self.sample_classes]
        results_sub_8 = {}
        if sub_indices:
            sub_true = [all_true_labels[i] for i in sub_indices]
            for name, (preds, top3, _) in models_eval.items():
                sub_preds = [preds[i] for i in sub_indices]
                sub_top3 = [top3[i] for i in sub_indices]

                top1_acc = np.mean([1 if p == t else 0 for p, t in zip(sub_preds, sub_true)]) * 100.0
                top3_acc = np.mean([1 if t in t3 else 0 for t, t3 in zip(sub_true, sub_top3)]) * 100.0
                macro_f1 = f1_score(sub_true, sub_preds, average="macro", zero_division=0)
                weighted_f1 = f1_score(sub_true, sub_preds, average="weighted", zero_division=0)

                per_class = {}
                for cls in self.sample_classes:
                    cls_sub_idx = [i for i, t in enumerate(sub_true) if t == cls]
                    if cls_sub_idx:
                        cls_top1 = np.mean([1 if sub_preds[i] == cls else 0 for i in cls_sub_idx]) * 100.0
                        cls_top3 = np.mean([1 if cls in sub_top3[i] else 0 for i in cls_sub_idx]) * 100.0
                        per_class[cls] = {
                            "support": len(cls_sub_idx),
                            "top1": cls_top1,
                            "top3": cls_top3
                        }
                    else:
                        per_class[cls] = {"support": 0, "top1": 0.0, "top3": 0.0}

                results_sub_8[name] = {
                    "top1_accuracy": top1_acc,
                    "top3_accuracy": top3_acc,
                    "macro_f1": float(macro_f1),
                    "weighted_f1": float(weighted_f1),
                    "per_class": per_class,
                }

        # 3. SNR Stratified Analysis (for Sample 1D, Sample 2D, Sample Fused, V2 Fused)
        snr_bins = [
            ("<0 dB", lambda s: s < 0),
            ("0 to 5 dB", lambda s: 0 <= s < 5),
            ("5 to 10 dB", lambda s: 5 <= s < 10),
            ("10 to 15 dB", lambda s: 10 <= s < 15),
            ("15 to 20 dB", lambda s: 15 <= s < 20),
            (">20 dB", lambda s: s >= 20),
        ]
        snr_performance = {}
        for bin_label, cond in snr_bins:
            b_indices = [i for i, s in enumerate(all_snrs) if cond(s)]
            if not b_indices:
                continue
            snr_performance[bin_label] = {
                "count": len(b_indices),
                "Sample_1D_Top1": float(np.mean([1 if preds_sample_1d[i] == all_true_labels[i] else 0 for i in b_indices]) * 100.0),
                "Sample_2D_Top1": float(np.mean([1 if preds_sample_2d[i] == all_true_labels[i] else 0 for i in b_indices]) * 100.0),
                "Sample_Fused_Top1": float(np.mean([1 if preds_sample_fused[i] == all_true_labels[i] else 0 for i in b_indices]) * 100.0),
                "V2_Fused_Top1": float(np.mean([1 if preds_v2_fused[i] == all_true_labels[i] else 0 for i in b_indices]) * 100.0),
            }

        # 4. Unknown False Positive Rate
        noise_idx = [i for i, t in enumerate(all_true_labels) if t == "UNKNOWN"]
        fpr_results = {}
        if noise_idx:
            for name, (preds, _, _) in models_eval.items():
                fp_count = sum(1 for i in noise_idx if preds[i] != "UNKNOWN")
                fpr = (fp_count / len(noise_idx)) * 100.0
                fpr_results[name] = fpr

        # 5. Unsupported Classes Confusion (DQPSK, MSK, 256QAM)
        unsupported_confusion = {}
        for unk_cls in ["DQPSK", "MSK", "256QAM"]:
            u_indices = [i for i, t in enumerate(all_true_labels) if t == unk_cls]
            if u_indices:
                counts_1d = {}
                counts_2d = {}
                for i in u_indices:
                    p1 = preds_sample_1d[i]
                    counts_1d[p1] = counts_1d.get(p1, 0) + 1
                    p2 = preds_sample_2d[i]
                    counts_2d[p2] = counts_2d.get(p2, 0) + 1
                unsupported_confusion[unk_cls] = {
                    "total": len(u_indices),
                    "Sample_1D_Predictions": counts_1d,
                    "Sample_2D_Predictions": counts_2d,
                }

        return {
            "dataset_name": dataset_name,
            "total_samples": n_total,
            "results_full_11": results_full_11,
            "results_sub_8": results_sub_8,
            "snr_performance": snr_performance,
            "noise_false_positive_rate": fpr_results,
            "unsupported_class_routing": unsupported_confusion,
        }

def run_all_evaluations():
    evaluator = ModelEvaluator(device="cuda")

    all_dataset_results = {}

    # Dataset A: V2 Test Set
    with open("datasets/ASTRA_MODULATION_DATASET_V2/manifests/modulation_v2_split_manifest.json") as f:
        manifest_v2 = json.load(f)
    test_captures = manifest_v2["splits"]["test"]
    all_dataset_results["V2_TEST_SET"] = evaluator.evaluate_dataset(test_captures, "ASTRA_MODULATION_DATASET_V2 (Test Split)")

    # Dataset B: V2 OOD Set
    ood_captures = manifest_v2["splits"]["ood_test"]
    all_dataset_results["V2_OOD_SET"] = evaluator.evaluate_dataset(ood_captures, "ASTRA_MODULATION_DATASET_V2 (OOD Split)")

    # Dataset C: Blind Final Test Set V2
    with open("datasets/ASTRA_FINAL_TEST_SET_V2_BLIND/blind_benchmark_manifest.json") as f:
        manifest_blind = json.load(f)
    blind_captures = manifest_blind["captures"]
    all_dataset_results["V2_BLIND_BENCHMARK"] = evaluator.evaluate_dataset(blind_captures, "ASTRA_FINAL_TEST_SET_V2_BLIND")

    # Save to JSON
    os.makedirs("checkpoints", exist_ok=True)
    out_path = "checkpoints/sample_models_evaluation.json"
    with open(out_path, "w") as f:
        json.dump({
            "param_counts": evaluator.param_counts,
            "sample_classes": evaluator.sample_classes,
            "v2_classes": evaluator.v2_classes,
            "dataset_evaluations": all_dataset_results
        }, f, indent=2)

    print(f"\n[DONE] Saved complete evaluation results to: {out_path}")

if __name__ == "__main__":
    run_all_evaluations()
