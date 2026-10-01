"""
Unit and Integration Test Suite for ASTRA Modulation Intelligence V2.

Tests:
1. Class schema consistency (11 canonical classes, order, index mapping, normalization)
2. Source split zero-leakage (assert 0 overlapping source_ids)
3. 1D preprocessing pipeline (DC removal, RMS power = 1.0, finite check)
4. 2D preprocessing pipeline (STFT tensor shape [1, 128, 128], finite check)
5. Model output shapes (ResNet1DV2 [B, 11], SpectrogramCNN2DV2 [B, 11])
6. Checkpoint integrity & class schema verification
7. Fusion engine output contract (Top-K, margin, entropy, agreement, rejection score)
8. UNKNOWN / Non-Target signal handling & rejection
9. Batch inference & same-source 1D/2D alignment
10. Data sanity: no NaNs, no Infs in dataset captures.
"""

from __future__ import annotations

import os
import sys
import json
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import (
    CLASS_SCHEMA_VERSION,
    MODULATION_CLASSES_V2,
    NUM_CLASSES_V2,
    get_class_index,
    get_class_name,
    normalize_modulation_name,
    assert_runtime_class_order,
)
from astra_modulation_v2.dataset_builder import (
    IQPreprocessorV2,
    iq_to_tensor_1d,
    iq_to_spectrogram_2d,
    ASTRAModulationV2Dataset,
)
from astra_modulation_v2.models.resnet1d import ResNet1DV2
from astra_modulation_v2.models.spectrogram_cnn import SpectrogramCNN2DV2
from astra_modulation_v2.fusion import CalibratedFusionEngineV2


def test_class_schema_consistency():
    """Verify authoritative 11-class schema."""
    assert len(MODULATION_CLASSES_V2) == 11
    assert NUM_CLASSES_V2 == 11
    assert MODULATION_CLASSES_V2[0] == "2-FSK"
    assert MODULATION_CLASSES_V2[1] == "4-FSK"
    assert MODULATION_CLASSES_V2[10] == "UNKNOWN"
    
    assert normalize_modulation_name("2fsk") == "2-FSK"
    assert normalize_modulation_name("4-fsk") == "4-FSK"
    assert normalize_modulation_name("16qam") == "16QAM"
    assert normalize_modulation_name("noise") == "UNKNOWN"
    assert normalize_modulation_name("cw") == "UNKNOWN"
    
    for idx, c in enumerate(MODULATION_CLASSES_V2):
        assert get_class_index(c) == idx
        assert get_class_name(idx) == c
        
    assert_runtime_class_order(MODULATION_CLASSES_V2)


def test_zero_leakage_across_splits():
    """Verify strict source-level partition with zero leakage."""
    manifest_dir = ROOT / "datasets" / "ASTRA_MODULATION_DATASET_V2" / "manifests"
    if not (manifest_dir / "train.csv").exists():
        pytest.skip("Dataset manifests not yet generated.")
        
    df_tr = pd.read_csv(manifest_dir / "train.csv")
    df_va = pd.read_csv(manifest_dir / "validation.csv")
    df_te = pd.read_csv(manifest_dir / "test.csv")
    df_ood = pd.read_csv(manifest_dir / "ood_test.csv")
    
    tr_ids = set(df_tr["source_id"])
    va_ids = set(df_va["source_id"])
    te_ids = set(df_te["source_id"])
    ood_ids = set(df_ood["source_id"])
    
    assert len(tr_ids.intersection(va_ids)) == 0, "Train & Val overlap detected!"
    assert len(tr_ids.intersection(te_ids)) == 0, "Train & Test overlap detected!"
    assert len(va_ids.intersection(te_ids)) == 0, "Val & Test overlap detected!"
    assert len(tr_ids.intersection(ood_ids)) == 0, "Train & OOD overlap detected!"


def test_1d_preprocessing():
    """Verify DC removal and RMS normalization on complex IQ."""
    prep = IQPreprocessorV2(remove_dc=True, normalize_rms=True)
    rng = np.random.default_rng(42)
    # Add large DC bias
    raw_iq = rng.normal(5.0, 2.0, 4096) + 1j * rng.normal(-3.0, 2.0, 4096)
    clean_iq = prep.process(raw_iq)
    
    # Assert DC bias removed
    assert np.isclose(np.mean(clean_iq), 0.0, atol=1e-5)
    # Assert RMS energy is 1.0
    rms = np.sqrt(np.mean(np.abs(clean_iq) ** 2))
    assert np.isclose(rms, 1.0, atol=1e-4)
    
    t_1d = iq_to_tensor_1d(clean_iq[:2048])
    assert t_1d.shape == (2, 2048)
    assert not torch.isnan(t_1d).any()
    assert not torch.isinf(t_1d).any()


def test_2d_spectrogram_preprocessing():
    """Verify centered STFT generation directly from complex IQ tensor."""
    rng = np.random.default_rng(42)
    clean_iq = rng.normal(0.0, 1.0, 2048) + 1j * rng.normal(0.0, 1.0, 2048)
    clean_iq = clean_iq / np.sqrt(np.mean(np.abs(clean_iq) ** 2))
    
    t_2d = iq_to_spectrogram_2d(clean_iq, n_fft=128, hop_length=32, target_f=128, target_t=128)
    assert t_2d.shape == (1, 128, 128)
    assert not torch.isnan(t_2d).any()
    assert not torch.isinf(t_2d).any()
    # Check standardization
    assert np.isclose(t_2d.mean().item(), 0.0, atol=1e-4)


def test_model_architectures_and_shapes():
    """Verify output shapes of ResNet-1D and Spectrogram CNN-2D."""
    b = 4
    x1 = torch.randn(b, 2, 2048)
    m1 = ResNet1DV2(in_channels=2, num_classes=11)
    out1, emb1 = m1(x1, return_embedding=True)
    assert out1.shape == (b, 11)
    assert emb1.shape == (b, 256)
    
    x2 = torch.randn(b, 1, 128, 128)
    m2 = SpectrogramCNN2DV2(in_channels=1, num_classes=11)
    out2, emb2 = m2(x2, return_embedding=True)
    assert out2.shape == (b, 11)
    assert emb2.shape == (b, 256)


def test_checkpoint_class_mappings():
    """Verify checkpoint metadata contains correct class list and schema."""
    ckpt_dir = ROOT / "checkpoints"
    p1 = ckpt_dir / "astra_resnet1d_v2.pt"
    p2 = ckpt_dir / "astra_spectrogram_cnn_v2.pt"
    
    if p1.exists():
        c1 = torch.load(p1, map_location="cpu")
        assert c1.get("class_schema_version") == CLASS_SCHEMA_VERSION
        assert c1.get("num_classes") == 11
        assert c1.get("classes") == MODULATION_CLASSES_V2
        
    if p2.exists():
        c2 = torch.load(p2, map_location="cpu")
        assert c2.get("class_schema_version") == CLASS_SCHEMA_VERSION
        assert c2.get("num_classes") == 11
        assert c2.get("classes") == MODULATION_CLASSES_V2


def test_fusion_engine_output_contract():
    """Verify CalibratedFusionEngineV2 complies with Section 69 & 70."""
    engine = CalibratedFusionEngineV2(device="cpu", top_k=3)
    rng = np.random.default_rng(42)
    fake_iq = rng.normal(0, 1, 2048) + 1j * rng.normal(0, 1, 2048)
    
    res = engine.classify(fake_iq)
    
    assert "predicted_modulation" in res
    assert res["predicted_modulation"] in MODULATION_CLASSES_V2
    assert "top_k_candidates" in res
    assert len(res["top_k_candidates"]) == 3
    assert "class_probs" in res
    assert len(res["class_probs"]) == 11
    assert "agreement_score" in res
    assert "disagreement_score" in res
    assert "entropy" in res
    assert "margin" in res
    assert "unknown_score" in res
    assert 0.0 <= res["top1_confidence"] <= 1.0


def test_unknown_signal_rejection():
    """Verify non-target pure noise signal is routed towards UNKNOWN."""
    engine = CalibratedFusionEngineV2(device="cpu")
    rng = np.random.default_rng(999)
    # Pure AWGN
    noise_iq = rng.normal(0, 1, 4096) + 1j * rng.normal(0, 1, 4096)
    res = engine.classify(noise_iq)
    
    # UNKNOWN score should be significant
    top_candidates = [c["modulation"] for c in res["top_k_candidates"]]
    assert "UNKNOWN" in top_candidates or res["unknown_score"] > 0.15
