import numpy as np
import torch

from astra_modulation_2d.src.dataset import SignalWindowRecord, ASTRASpectrogramDataset, create_2d_dataloaders
from astra_modulation_2d.src.preprocessing import IQPreprocessor
from astra_modulation_2d.src.spectrogram import SpectrogramGenerator


def test_dataset_item_generation():
    """Verifies that dataset properly outputs [1, F, T] tensors with accurate labels."""
    classes = ["bpsk", "qpsk"]
    records = []
    for i in range(10):
        # Create dummy cached IQ
        dummy_iq = np.random.randn(8192) + 1j * np.random.randn(8192)
        records.append(SignalWindowRecord(
            source="synthetic",
            source_signal_id=f"dummy_{i}",
            modulation=classes[i % 2],
            target=i % 2,
            snr_db=15.0,
            start_sample=0,
            window_size=2048,
            cached_iq=dummy_iq,
        ))

    ds = ASTRASpectrogramDataset(records, classes)
    assert len(ds) == 10

    spec, target, snr, sid, start_s = ds[0]
    assert spec.dim() == 3  # [1, 128, 65]
    assert spec.shape[0] == 1
    assert spec.shape[1] == 128
    assert spec.shape[2] == 65
    assert target in [0, 1]
    assert snr == 15.0
    assert sid == "dummy_0"
    assert start_s == 0
