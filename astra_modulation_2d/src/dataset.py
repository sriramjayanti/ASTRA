"""
ASTRA 2D Spectrogram Dataset Loader.

Supports:
- Mode A: CSPB.ML.2018R2 (28 Batches, .tim streaming with bad signal filtering)
- Mode B: ASTRA Synthetic Datasets (2-FSK, 4-FSK, framed, FEC-encoded, interleaved captures)
- On-the-fly GPU/CPU STFT computation
- Window slicing strictly after source signal partition
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import IQPreprocessor
from .spectrogram import SpectrogramGenerator
from .splits import load_split_json, save_split_json, verify_disjoint_splits

logger = logging.getLogger("ASTRA_DATASET_2D")

CSPB_COLUMNS = [
    "signal_index", "modulation", "t0", "carrier_offset",
    "rolloff", "u", "d", "snr_db", "noise_density_db"
]


class SignalWindowRecord:
    """Metadata for a single window extracted from a source recording."""
    def __init__(
        self,
        source: str,
        source_signal_id: str,
        modulation: str,
        target: int,
        snr_db: float,
        start_sample: int,
        window_size: int,
        # For synthetic files:
        filepath: Optional[Path] = None,
        # For CSPB zip entries:
        zip_path: Optional[str] = None,
        internal_path: Optional[str] = None,
        cached_iq: Optional[np.ndarray] = None,
    ):
        self.source = source
        self.source_signal_id = str(source_signal_id)
        self.modulation = modulation.lower().strip()
        self.target = target
        self.snr_db = float(snr_db)
        self.start_sample = start_sample
        self.window_size = window_size
        self.filepath = filepath
        self.zip_path = zip_path
        self.internal_path = internal_path
        self.cached_iq = cached_iq


class ASTRASpectrogramDataset(Dataset):
    """
    PyTorch Dataset for 2D Spectrogram Generation from Complex Baseband Signals.
    """
    def __init__(
        self,
        records: List[SignalWindowRecord],
        class_names: List[str],
        preprocessor: Optional[IQPreprocessor] = None,
        spectrogram_generator: Optional[SpectrogramGenerator] = None,
        cache_spectrograms: bool = False,
    ):
        self.records = records
        self.class_names = [c.lower().strip() for c in class_names]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.preprocessor = preprocessor or IQPreprocessor()
        self.spectrogram_generator = spectrogram_generator or SpectrogramGenerator()
        self.cache_spectrograms = cache_spectrograms
        self._spec_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_raw_iq(self, rec: SignalWindowRecord) -> np.ndarray:
        if rec.cached_iq is not None:
            raw = rec.cached_iq
            start_s = rec.start_sample
            w_size = rec.window_size
            if len(raw) >= start_s + w_size:
                return raw[start_s : start_s + w_size]
            else:
                out = np.zeros(w_size, dtype=np.complex64)
                avail = min(len(raw), w_size)
                out[:avail] = raw[:avail]
                return out

        if rec.source == "cspb":
            with zipfile.ZipFile(rec.zip_path, "r") as zf:
                raw_bytes = zf.read(rec.internal_path)
            scalars = np.frombuffer(raw_bytes, dtype=np.float32)
            start_s = rec.start_sample * 2
            end_s = start_s + (rec.window_size * 2)
            if end_s <= len(scalars):
                w_scalars = scalars[start_s:end_s]
            else:
                w_scalars = scalars[: rec.window_size * 2]
            i_ch = w_scalars[0::2]
            q_ch = w_scalars[1::2]
            return (i_ch + 1j * q_ch).astype(np.complex64)

        elif rec.source == "synthetic":
            # Load synthetic .iq / .wav file
            p = rec.filepath
            if p.suffix.lower() == ".wav":
                from scipy.io import wavfile
                _, data = wavfile.read(str(p))
                if data.ndim == 2:
                    i_s, q_s = data[:, 0].astype(np.float32), data[:, 1].astype(np.float32)
                else:
                    i_s, q_s = data[0::2].astype(np.float32), data[1::2].astype(np.float32)
                if np.issubdtype(data.dtype, np.integer):
                    max_v = float(np.iinfo(data.dtype).max)
                    i_s /= max_v
                    q_s /= max_v
                raw = (i_s + 1j * q_s).astype(np.complex64)
            else:
                # Binary .iq file
                # Check companion capture.json if available
                cap_json = p.parent.parent / "visible_metadata" / f"{p.stem}.capture.json"
                dtype_str = "float32"
                if cap_json.exists():
                    try:
                        with open(cap_json, "r") as f:
                            meta = json.load(f)
                        dtype_str = meta.get("storage_dtype", "float32")
                    except Exception:
                        pass
                np_dt = np.int16 if dtype_str == "int16" else (np.int8 if dtype_str == "int8" else np.float32)
                arr = np.fromfile(str(p), dtype=np_dt)
                if len(arr) % 2 != 0:
                    arr = arr[:-1]
                i_s = arr[0::2].astype(np.float32)
                q_s = arr[1::2].astype(np.float32)
                if np.issubdtype(np_dt, np.integer):
                    max_v = float(np.iinfo(np_dt).max)
                    i_s /= max_v
                    q_s /= max_v
                raw = (i_s + 1j * q_s).astype(np.complex64)

            rec.cached_iq = raw
            start_s = rec.start_sample
            w_size = rec.window_size
            if len(raw) >= start_s + w_size:
                return raw[start_s : start_s + w_size]
            else:
                out = np.zeros(w_size, dtype=np.complex64)
                avail = min(len(raw), w_size)
                out[:avail] = raw[:avail]
                return out

        return np.zeros(rec.window_size, dtype=np.complex64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, str, int]:
        """
        Returns:
            spectrogram: [1, Freq_Bins, Time_Frames]
            target: int class index
            snr_db: float SNR
            source_signal_id: str
            start_sample: int
        """
        if self.cache_spectrograms and idx in self._spec_cache:
            spec = self._spec_cache[idx]
        else:
            rec = self.records[idx]
            raw_complex = self._load_raw_iq(rec)
            clean_complex = self.preprocessor(raw_complex)

            # Convert to PyTorch tensor [2, N] for SpectrogramGenerator
            i_ch = np.real(clean_complex).astype(np.float32)
            q_ch = np.imag(clean_complex).astype(np.float32)
            iq_2ch = torch.from_numpy(np.stack([i_ch, q_ch], axis=0)).unsqueeze(0)  # [1, 2, N]

            with torch.no_grad():
                spec = self.spectrogram_generator(iq_2ch).squeeze(0)  # [1, F, T]

            if self.cache_spectrograms:
                self._spec_cache[idx] = spec

        rec = self.records[idx]
        return spec, rec.target, rec.snr_db, rec.source_signal_id, rec.start_sample


def create_2d_dataloaders(
    train_records: List[SignalWindowRecord],
    val_records: List[SignalWindowRecord],
    test_records: List[SignalWindowRecord],
    class_names: List[str],
    batch_size: int = 64,
    preprocessor: Optional[IQPreprocessor] = None,
    spectrogram_generator: Optional[SpectrogramGenerator] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders for Train, Val, and Test splits."""
    train_ds = ASTRASpectrogramDataset(train_records, class_names, preprocessor, spectrogram_generator)
    val_ds = ASTRASpectrogramDataset(val_records, class_names, preprocessor, spectrogram_generator)
    test_ds = ASTRASpectrogramDataset(test_records, class_names, preprocessor, spectrogram_generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
