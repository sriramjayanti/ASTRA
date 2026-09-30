"""
ASTRA End-to-End Pipeline Execution:
1. Finish Synthetic Engines Verification
2. Generate 100 Test Signals
3. Validate Complete Ground Truth
4. Generate Training Dataset
5. Train the Ready 1D ResNet Model
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Import Synthetic Engines (Engine 8 Orchestration)
from astra_synthetic.orchestration.generator import ASTRASyntheticDatasetGenerator
from astra_synthetic.orchestration.splits import verify_no_split_leakage

# Import 1D ResNet Model & Dataset Components
import sys
starter_src = Path(__file__).parent.parent.parent / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "src"
if str(starter_src) not in sys.path:
    sys.path.insert(0, str(starter_src))

from model import ResNet1DModClassifier
from dataset import (
    load_truth_metadata,
    create_file_level_splits,
    get_dataloader,
    IQPreprocessor,
    read_tim_file,
    complex_to_channels,
)
from inference import predict_modulation

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ASTRA_PIPELINE")


def main():
    workspace_root = Path(__file__).parent.parent.parent
    test_100_dir = workspace_root / "datasets" / "test_100_signals"
    train_ds_dir = workspace_root / "datasets" / "astra_training_dataset"
    ckpt_dir = workspace_root / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "checkpoints"
    eval_dir = workspace_root / "ASTRA_Modulation_Model_Starter" / "ASTRA_Modulation_Model_Starter" / "eval_results"
    
    test_100_dir.mkdir(parents=True, exist_ok=True)
    train_ds_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ASTRA: SYNTHETIC ENGINES -> 100 TEST SIGNALS -> GROUND TRUTH -> TRAIN 1D MODEL")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1 & 2: Generate 100 Test Signals using Synthetic Engine 8
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> STAGE 1 & 2: GENERATING 100 TEST SIGNALS ACROSS FULL SYNTHETIC PIPELINE")
    print("-" * 80)
    
    config_path = workspace_root / "astra_synthetic" / "configs" / "dataset_config.yaml"
    orchestrator_100 = ASTRASyntheticDatasetGenerator(config=config_path)
    
    test_records_100 = orchestrator_100.generate_dataset(
        count=100,
        output_root=test_100_dir,
        show_progress=True,
    )
    print(f"[OK] Generated {len(test_records_100)} physical signal captures in {test_100_dir}")

    # -------------------------------------------------------------------------
    # STAGE 3: Validate Complete Ground Truth & Split Leakage
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> STAGE 3: VALIDATING COMPLETE GROUND TRUTH & ZERO LEAKAGE")
    print("-" * 80)

    valid_count = sum(1 for r in test_records_100 if r.valid)
    print(f"  * Total Records Validated: {valid_count} / {len(test_records_100)} ({valid_count/len(test_records_100)*100:.1f}%)")

    # Check Source-Chain ID Lineage
    first_rec = test_records_100[0]
    print(f"  * Sample Lineage Audit for {first_rec.dataset_record_id}:")
    print(f"    - Source Chain ID: {first_rec.source_chain_id}")
    print(f"    - Payload ID:      {first_rec.payload_id}")
    print(f"    - Frame ID:        {first_rec.frame_id}")
    print(f"    - FEC Record ID:   {first_rec.fec_record_id}")
    print(f"    - Interleaver ID:  {first_rec.interleaver_record_id}")
    print(f"    - Modulation ID:   {first_rec.modulation_record_id}")
    print(f"    - Channel ID:      {first_rec.channel_record_id}")
    print(f"    - Capture ID:      {first_rec.capture_record_id}")
    print(f"    - Physical Path:   {first_rec.capture_path}")
    print(f"    - Truth JSON:      {first_rec.truth_metadata_path}")
    print(f"    - Visible JSON:    {first_rec.visible_metadata_path}")

    # Verify Split Leakage
    is_disjoint, overlap_report = verify_no_split_leakage(test_records_100)
    if is_disjoint:
        print("  [OK] PASSED: ZERO split leakage detected across Train/Val/Test source chains!")
    else:
        print(f"  [FAIL] Leakage detected: {overlap_report}")
        raise RuntimeError("Split leakage in 100 test dataset!")

    # Verify Visible Metadata hides ground truth
    with open(first_rec.visible_metadata_path, "r", encoding="utf-8") as f:
        vis_meta = json.load(f)
    forbidden = ["modulation", "payload_bits", "fec_type", "cfo_hz", "snr_db"]
    leaks = [k for k in forbidden if k in vis_meta]
    if not leaks:
        print("  [OK] PASSED: Visible metadata (.capture.json) contains ZERO hidden ground-truth labels!")
    else:
        print(f"  [FAIL] Label leakage in visible metadata: {leaks}")

    # -------------------------------------------------------------------------
    # STAGE 4: Generate Training Dataset
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> STAGE 4: GENERATING DEDICATED SYNTHETIC TRAINING DATASET (320 SIGNALS)")
    print("-" * 80)

    train_signal_count = 320
    orchestrator_train = ASTRASyntheticDatasetGenerator(config=config_path)
    train_records = orchestrator_train.generate_dataset(
        count=train_signal_count,
        output_root=train_ds_dir,
        show_progress=True,
    )
    print(f"[OK] Generated {len(train_records)} training signals in {train_ds_dir}")

    # Check generated manifests
    mod_manifest_path = train_ds_dir / "manifests" / "modulation.csv"
    print(f"  * Training Modulation Manifest: {mod_manifest_path}")

    # -------------------------------------------------------------------------
    # STAGE 5: Train the Ready 1D ResNet Model
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> STAGE 5: TRAINING 1D RESNET MODEL ON GENERATED SYNTHETIC DATASET")
    print("-" * 80)

    # 1. Target classes
    target_classes = ["2fsk", "4fsk", "bpsk", "qpsk", "8psk", "16qam", "64qam", "msk"]
    print(f"  * Training Classes ({len(target_classes)}): {target_classes}")

    # 2. Load manifest as truth dataframe
    truth_df = load_truth_metadata(mod_manifest_path)
    print(f"  * Loaded {len(truth_df)} total records from modulation manifest")

    # 3. File-level splits
    train_df, val_df, test_df = create_file_level_splits(
        data_dir=train_ds_dir / "captures",
        truth_df=truth_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        output_dir=ckpt_dir / "manifests",
        stratify_by_mod=True,
    )
    print(f"  * Split: {len(train_df)} Train files | {len(val_df)} Val files | {len(test_df)} Test files")

    # 4. Preprocessor & DataLoaders
    preprocessor = IQPreprocessor(remove_dc=True, normalization="rms")
    window_size = 2048

    train_loader = get_dataloader(
        manifest_df=train_df,
        data_dir=train_ds_dir / "captures",
        tim_dtype="float32",
        classes=target_classes,
        batch_size=32,
        shuffle=True,
        window_size=window_size,
        preprocessor=preprocessor,
    )

    val_loader = get_dataloader(
        manifest_df=val_df,
        data_dir=train_ds_dir / "captures",
        tim_dtype="float32",
        classes=target_classes,
        batch_size=32,
        shuffle=False,
        window_size=window_size,
        preprocessor=preprocessor,
    )

    test_loader = get_dataloader(
        manifest_df=test_df,
        data_dir=train_ds_dir / "captures",
        tim_dtype="float32",
        classes=target_classes,
        batch_size=32,
        shuffle=False,
        window_size=window_size,
        preprocessor=preprocessor,
    )

    print(f"  * Windows: {len(train_loader.dataset)} Train | {len(val_loader.dataset)} Val | {len(test_loader.dataset)} Test")

    # 5. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  * Target Compute Device: {device}")

    model = ResNet1DModClassifier(
        num_classes=len(target_classes),
        input_channels=2,
        base_channels=64,
        dropout=0.2,
    ).to(device)
    print(f"  * 1D ResNet Architecture Initialized ({model.count_parameters():,} trainable parameters)")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 12
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. Training Loop
    best_f1 = -1.0
    best_model_path = ckpt_dir / "best_model_synthetic.pt"

    print(f"\n--- Starting 1D ResNet Training ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(by.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        scheduler.step()

        dur = time.time() - t0
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({dur:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": float(val_acc * 100),
                "val_macro_f1": float(val_f1),
                "classes": target_classes,
                "config": {
                    "classes": target_classes,
                    "window_size": window_size,
                    "model_type": "resnet1d",
                }
            }, best_model_path)
            print(f"    -> Saved new best checkpoint (Val F1: {val_f1:.4f})")

    # -------------------------------------------------------------------------
    # STAGE 6: Evaluation & Sample Inference
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> STAGE 6: TEST EVALUATION & INFERENCE SANITY CHECK")
    print("-" * 80)

    # Load best checkpoint
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_preds, test_targets, test_snrs = [], [], []
    with torch.no_grad():
        for bx, by, b_snr in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_targets.extend(by.numpy())
            test_snrs.extend(b_snr.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    prec, rec, f1_per_cls, supp = precision_recall_fscore_support(
        test_targets, test_preds, labels=range(len(target_classes)), average=None, zero_division=0
    )

    print(f"\n[FINAL TEST METRICS]")
    print(f"  • Overall Test Accuracy: {test_acc*100:.2f}%")
    print(f"  • Macro-F1 Score:        {test_f1:.4f}")
    print(f"\n  [Per-Class Breakdown]")
    for i, cname in enumerate(target_classes):
        print(f"    - {cname:10s}: Precision: {prec[i]*100:5.1f}% | Recall: {rec[i]*100:5.1f}% | F1: {f1_per_cls[i]:.4f} (n={supp[i]})")

    # Sample Inference API demonstration
    print(f"\n[SAMPLE INFERENCE TEST (Top-K Probabilities)]")
    sample_iq_file = test_df.iloc[0]["resolved_file_path"]
    raw_iq_test = read_tim_file(sample_iq_file, dtype="float32")[:window_size]
    
    inf_result = predict_modulation(
        raw_iq_test,
        model=model,
        classes=target_classes,
        preprocessor=preprocessor,
        top_k=3,
        device=device,
    )
    print(f"  * Sample File: {Path(sample_iq_file).name}")
    print(f"  * Ground Truth Modulation: {test_df.iloc[0]['modulation']}")
    print(f"  * Predicted Modulation:    {inf_result['predicted_class']} (Confidence: {inf_result['confidence']*100:.2f}%)")
    print(f"  * Top-3 Candidates:        {inf_result['top_k']}")

    print("\n" + "=" * 80)
    print("ALL REQUESTED STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
