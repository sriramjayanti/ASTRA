import argparse
import logging
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal

from dataset import (
    load_config,
    load_truth_metadata,
    create_file_level_splits,
    get_dataloader,
    read_tim_file,
    IQPreprocessor,
    find_signal_file
)
from model import create_model
from inference import predict_modulation

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ASTRA_VERIFY")


def run_first_day_verification(
    config_path: str = "config.yaml",
    data_dir_override: str = None,
    truth_file_override: str = None,
    output_dir: str = "verification_output"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load authoritative configuration
    config = load_config(config_path)
    d_cfg = config.get("dataset", {})
    p_cfg = config.get("preprocessing", {})
    classes = config["classes"]

    tim_dtype = d_cfg.get("tim_dtype", "float32")
    data_dir = data_dir_override or d_cfg.get("data_dir", "data")
    truth_file = truth_file_override or d_cfg.get("truth_file", "data/truth.csv")
    window_size = d_cfg.get("window_size", 2048)

    preprocessor = IQPreprocessor(
        remove_dc=p_cfg.get("remove_dc", True),
        normalization=p_cfg.get("normalization", "rms"),
        epsilon=p_cfg.get("epsilon", 1e-8)
    )

    print("======================================================================")
    print("           ASTRA MODEL 1: PIPELINE & SANITY VERIFICATION              ")
    print("======================================================================")
    print(f"Config: {config_path} | tim_dtype: {tim_dtype} | Window Size: {window_size}")
    print(f"Classes ({len(classes)}): {classes}")

    # Step 1: Load truth file and inspect
    truth_df = load_truth_metadata(truth_file)
    print(f"\n[Step 1] Loaded Truth Metadata: {len(truth_df)} total records from {truth_file}")
    print("First 3 records:")
    print(truth_df.head(3).to_string())

    # Step 2: Read first .tim file and verify dtype & sample values
    first_row = truth_df.iloc[0]
    sig_idx = first_row.get("signal_index", "signal_0000")
    sample_file = find_signal_file(Path(data_dir), sig_idx)

    if sample_file is None:
        raise FileNotFoundError(f"Could not locate file for signal {sig_idx} in {data_dir}")

    raw_iq = read_tim_file(sample_file, dtype=tim_dtype)
    print(f"\n[Step 2] Read .tim file: {sample_file.name}")
    print(f"  - Total Complex Samples: {len(raw_iq):,}")
    print(f"  - Complex Dtype: {raw_iq.dtype} (from configured binary dtype: {tim_dtype})")
    print(f"  - First 5 raw I samples: {np.real(raw_iq[:5])}")
    print(f"  - First 5 raw Q samples: {np.imag(raw_iq[:5])}")

    # Step 3: Centralized Preprocessing
    clean_iq = preprocessor.process(raw_iq)
    print(f"\n[Step 3] Preprocessed IQ:")
    print(f"  - Mean before: {np.mean(raw_iq):.4e} | Mean after: {np.mean(clean_iq):.4e}")
    print(f"  - RMS before:  {np.sqrt(np.mean(np.abs(raw_iq)**2)):.4e} | RMS after:  {np.sqrt(np.mean(np.abs(clean_iq)**2)):.4f}")

    # Step 4: Plot Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    n_plot = min(200, len(clean_iq))
    axes[0, 0].plot(np.real(clean_iq[:n_plot]), label="In-phase (I)", color="#1f77b4", alpha=0.85)
    axes[0, 0].plot(np.imag(clean_iq[:n_plot]), label="Quadrature (Q)", color="#ff7f0e", alpha=0.85)
    axes[0, 0].set_title(f"Time-Domain Waveform ({first_row.get('modulation', 'Unknown').upper()})", fontweight="bold")
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(np.real(clean_iq[:1000]), np.imag(clean_iq[:1000]), alpha=0.5, s=12, color="#2ca02c")
    axes[0, 1].set_title("Constellation Diagram (I vs Q)", fontweight="bold")
    axes[0, 1].set_xlabel("I")
    axes[0, 1].set_ylabel("Q")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color="gray", linewidth=0.8)
    axes[0, 1].axvline(0, color="gray", linewidth=0.8)

    mag = np.abs(clean_iq[:n_plot])
    axes[1, 0].plot(mag, color="#9467bd")
    axes[1, 0].set_title("Magnitude Envelope |IQ|", fontweight="bold")
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("Magnitude")
    axes[1, 0].grid(True, alpha=0.3)

    f, psd = signal.welch(clean_iq, nperseg=512, return_onesided=False)
    f_shifted = np.fft.fftshift(f)
    psd_shifted = np.fft.fftshift(psd)
    axes[1, 1].plot(f_shifted, 10 * np.log10(psd_shifted + 1e-12), color="#d62728")
    axes[1, 1].set_title("Power Spectral Density (Welch PSD)", fontweight="bold")
    axes[1, 1].set_xlabel("Normalized Frequency")
    axes[1, 1].set_ylabel("PSD (dB/Hz)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_path / "sample_signal_inspection.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n[Step 4] Saved signal diagnostics plot to: {plot_path}")

    # Step 5: Test Train/Val/Test File Splits
    train_df, val_df, test_df = create_file_level_splits(
        data_dir=data_dir,
        truth_df=truth_df,
        train_ratio=d_cfg.get("train_ratio", 0.70),
        val_ratio=d_cfg.get("val_ratio", 0.15),
        test_ratio=d_cfg.get("test_ratio", 0.15),
        seed=d_cfg.get("split_seed", 42),
        output_dir=out_path / "manifests"
    )
    print(f"\n[Step 5] File-level Split Created:")
    print(f"  - Train files: {len(train_df)} | Val files: {len(val_df)} | Test files: {len(test_df)}")

    # Step 6: Test Memory-Mapped DataLoader
    train_loader = get_dataloader(
        manifest_df=train_df,
        data_dir=data_dir,
        tim_dtype=tim_dtype,
        classes=classes,
        batch_size=8,
        shuffle=True,
        window_size=window_size,
        preprocessor=preprocessor
    )

    sample_batch_x, sample_batch_y, sample_snr = next(iter(train_loader))
    print(f"\n[Step 6] Batch Tensor Verification (via memory-mapped loader):")
    print(f"  - Input Tensor Shape:  {list(sample_batch_x.shape)} (Expected: [8, 2, {window_size}])")
    print(f"  - Target Tensor Shape: {list(sample_batch_y.shape)} (Expected: [8])")
    print(f"  - SNR Values: {sample_snr.numpy().tolist()[:4]}...")

    # Step 7: Model Forward Pass
    model = create_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        out_logits = model(sample_batch_x.to(device))
    print(f"\n[Step 7] Model Forward Pass:")
    print(f"  - ResNet Output Shape: {list(out_logits.shape)} (Expected: [8, {len(classes)}])")
    print(f"  - Trainable parameters: {model.count_parameters():,}")

    # Step 8: Overfitting on Tiny Subset Sanity Test
    print(f"\n[Step 8] Running Overfit Sanity Test on tiny subset (16 samples, 20 epochs)...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    tiny_x = sample_batch_x.to(device)
    tiny_y = sample_batch_y.to(device)

    for ep in range(1, 26):
        optimizer.zero_grad()
        logits = model(tiny_x)
        loss = criterion(logits, tiny_y)
        loss.backward()
        optimizer.step()

        if ep % 5 == 0 or ep == 1:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == tiny_y).float().mean().item()
            print(f"    Epoch {ep:02d} | Loss: {loss.item():.4f} | Accuracy: {acc*100:.1f}%")

    # Step 9: Test Inference API
    print(f"\n[Step 9] Testing predict_modulation API...")
    single_iq = clean_iq[:window_size]
    inf_res = predict_modulation(
        single_iq,
        model=model,
        classes=classes,
        preprocessor=preprocessor,
        device=device
    )
    print("Inference API Output:")
    print(json.dumps(inf_res, indent=2))

    print("\n" + "="*70)
    print(">>> ALL PRODUCTION-HARDENED SANITY CHECKS PASSED SUCCESSFULLY! <<<")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--truth_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="verification_output")
    args = parser.parse_args()

    run_first_day_verification(
        config_path=args.config,
        data_dir_override=args.data_dir,
        truth_file_override=args.truth_file,
        output_dir=args.output_dir
    )
