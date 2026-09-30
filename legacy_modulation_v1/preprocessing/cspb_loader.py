"""
Unified High-Performance CSPB.ML.2018R2 Dataset Loader for ASTRA.

Capabilities:
1. Indexes all 28 CSPB zip archives (111,999 signal files) and maps to truth.txt.
2. Fast direct zip streaming / extraction with in-memory caching.
3. Strict file-level train/val/test splitting with zero signal leakage.
4. Windowing (default N=2048) strictly after file splitting.
5. Centralized IQPreprocessor (DC removal + RMS normalization).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("CSPB_LOADER")

CSPB_CLASSES = ["bpsk", "qpsk", "8psk", "dqpsk", "msk", "16qam", "64qam", "256qam"]
CSPB_COLUMNS = [
    "signal_index",
    "modulation",
    "t0",
    "carrier_offset",
    "rolloff",
    "u",
    "d",
    "snr_db",
    "noise_density_db",
]


class IQPreprocessor:
    """Centralized preprocessor for complex IQ samples."""
    def __init__(self, remove_dc: bool = True, normalization: str = "rms", epsilon: float = 1e-8):
        self.remove_dc = remove_dc
        self.normalization = normalization.lower().strip()
        self.epsilon = float(epsilon)

    def process(self, iq_samples: np.ndarray) -> np.ndarray:
        iq_samples = np.nan_to_num(iq_samples, nan=0.0, posinf=0.0, neginf=0.0)
        if self.remove_dc and len(iq_samples) > 0:
            m = np.mean(iq_samples)
            if np.isfinite(m):
                iq_samples = iq_samples - m
        if self.normalization == "rms" and len(iq_samples) > 0:
            rms = np.sqrt(np.mean(np.abs(iq_samples) ** 2))
            if np.isfinite(rms) and rms > self.epsilon:
                iq_samples = iq_samples / rms
            else:
                iq_samples = np.zeros_like(iq_samples)
        elif self.normalization == "peak" and len(iq_samples) > 0:
            peak = np.max(np.abs(iq_samples))
            if np.isfinite(peak) and peak > self.epsilon:
                iq_samples = iq_samples / peak
        return np.nan_to_num(iq_samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)


def complex_to_channels(complex_window: np.ndarray) -> np.ndarray:
    """Converts 1D complex IQ window [N] to 2D real tensor [2, N]."""
    i_ch = np.real(complex_window).astype(np.float32)
    q_ch = np.imag(complex_window).astype(np.float32)
    return np.stack([i_ch, q_ch], axis=0)


class CSPBDatasetCatalog:
    """Indexes all 28 CSPB zip archives and truth metadata."""
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.truth_file = self.data_dir / "truth.txt"
        self.sig_index_map: Dict[int, Tuple[Path, str]] = {}
        self.truth_df: Optional[pd.DataFrame] = None
        self._build_index()

    def _build_index(self):
        logger.info(f"Indexing CSPB dataset from: {self.data_dir}")
        if not self.truth_file.exists():
            raise FileNotFoundError(f"truth.txt not found at: {self.truth_file}")

        # 1. Load truth.txt
        self.truth_df = pd.read_csv(self.truth_file, sep=r"\s+", header=None, names=CSPB_COLUMNS)
        self.truth_df["modulation"] = self.truth_df["modulation"].astype(str).str.lower().str.strip()

        # Load bad signals blacklist if available
        bad_signals = set()
        bad_list_file = self.data_dir / "bad_signals_list.json"
        if bad_list_file.exists():
            try:
                with open(bad_list_file, "r") as f:
                    bad_signals = set(json.load(f))
                logger.info(f"Loaded {len(bad_signals):,} blacklisted corrupt entries from bad_signals_list.json")
            except Exception as e:
                logger.warning(f"Failed loading bad_signals_list.json: {e}")

        # 2. Index all 28 zip archives
        for b_idx in range(1, 29):
            zpath = self.data_dir / f"CSPB.ML_.2018R2_{b_idx}.zip"
            if not zpath.exists():
                logger.warning(f"Zip archive missing: {zpath.name}")
                continue
            with zipfile.ZipFile(zpath, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".tim") and name not in bad_signals:
                        stem = Path(name).stem
                        sig_num = int(stem.replace("signal_", ""))
                        self.sig_index_map[sig_num] = (zpath, name)

        logger.info(f"Successfully indexed {len(self.sig_index_map):,} verified signal files across 28 archives.")


def create_cspb_file_splits(
    catalog: CSPBDatasetCatalog,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_signals_per_class: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates file-level train/val/test splits across the complete 28-batch CSPB dataset.
    Guarantees zero signal leakage.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    rng = np.random.default_rng(seed)

    # Filter truth_df to only available files
    available_rows = []
    for idx, row in catalog.truth_df.iterrows():
        sig_id = int(row["signal_index"])
        if sig_id in catalog.sig_index_map:
            zpath, internal_path = catalog.sig_index_map[sig_id]
            rdict = row.to_dict()
            rdict["zip_path"] = str(zpath)
            rdict["internal_path"] = internal_path
            available_rows.append(rdict)

    df = pd.DataFrame(available_rows)
    logger.info(f"Verified {len(df):,} total available signals in truth catalog.")

    train_indices, val_indices, test_indices = [], [], []

    for mod, group in df.groupby("modulation"):
        idx_list = group.index.to_numpy()
        rng.shuffle(idx_list)
        if max_signals_per_class is not None and max_signals_per_class < len(idx_list):
            idx_list = idx_list[:max_signals_per_class]

        n_total = len(idx_list)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        train_indices.extend(idx_list[:n_train])
        val_indices.extend(idx_list[n_train : n_train + n_val])
        test_indices.extend(idx_list[n_train + n_val :])

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    # Assert zero leakage
    train_sigs = set(train_df["signal_index"])
    val_sigs = set(val_df["signal_index"])
    test_sigs = set(test_df["signal_index"])

    assert len(train_sigs.intersection(val_sigs)) == 0, "Leakage between Train and Val!"
    assert len(train_sigs.intersection(test_sigs)) == 0, "Leakage between Train and Test!"
    assert len(val_sigs.intersection(test_sigs)) == 0, "Leakage between Val and Test!"

    logger.info(f"Split Created (ZERO LEAKAGE): {len(train_df):,} Train | {len(val_df):,} Val | {len(test_df):,} Test signals.")
    return train_df, val_df, test_df


class CSPBStreamingDataset(Dataset):
    """
    Streaming PyTorch Dataset reading directly from CSPB zip archives.
    Extracts windows of size N=2048 with centralized preprocessing.
    """
    def __init__(
        self,
        split_df: pd.DataFrame,
        classes: List[str] = CSPB_CLASSES,
        window_size: int = 2048,
        stride: Optional[int] = 2048,
        windows_per_signal: int = 4,
        preprocessor: Optional[IQPreprocessor] = None,
        preload_memory: bool = True,
    ):
        self.split_df = split_df
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.window_size = window_size
        self.stride = stride or window_size
        self.windows_per_signal = windows_per_signal
        self.preprocessor = preprocessor or IQPreprocessor()
        self.preload_memory = preload_memory

        self.samples: List[dict] = []
        self._signal_cache: Dict[str, np.ndarray] = {}
        self._index_windows()
        if self.preload_memory:
            self._preload_all_signals()

    def _index_windows(self):
        for _, row in self.split_df.iterrows():
            mod = row["modulation"]
            if mod not in self.class_to_idx:
                continue
            target = self.class_to_idx[mod]
            snr = float(row["snr_db"])
            zip_p = row["zip_path"]
            internal_p = row["internal_path"]
            sig_id = row["signal_index"]

            # 32,769 complex samples per file -> create windows_per_signal windows
            for w in range(self.windows_per_signal):
                start_s = w * self.stride
                self.samples.append({
                    "zip_path": zip_p,
                    "internal_path": internal_p,
                    "start_sample": start_s,
                    "target": target,
                    "modulation": mod,
                    "snr_db": snr,
                    "signal_index": sig_id,
                })

        logger.info(f"Indexed {len(self.samples):,} windows (size={self.window_size}) across {len(self.split_df):,} signal files.")

    def _preload_all_signals(self):
        logger.info(f"Preloading {len(self.split_df):,} signal files into memory cache...")
        # Group by zip path to open each zip file only once
        grouped = self.split_df.groupby("zip_path")
        for zip_p, group in grouped:
            try:
                with zipfile.ZipFile(zip_p, "r") as zf:
                    for _, row in group.iterrows():
                        internal_p = row["internal_path"]
                        if internal_p not in self._signal_cache:
                            raw_bytes = zf.read(internal_p)
                            self._signal_cache[internal_p] = np.frombuffer(raw_bytes, dtype=np.float32)
            except Exception as e:
                logger.error(f"Error preloading from zip {zip_p}: {e}")
        logger.info(f"Preloaded {len(self._signal_cache):,} signals into memory.")

    def _get_signal_scalars(self, zip_path: str, internal_path: str) -> np.ndarray:
        if internal_path in self._signal_cache:
            return self._signal_cache[internal_path]
        with zipfile.ZipFile(zip_path, "r") as zf:
            raw_bytes = zf.read(internal_path)
            scalars = np.frombuffer(raw_bytes, dtype=np.float32)
        self._signal_cache[internal_path] = scalars
        return scalars

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        info = self.samples[idx]
        raw_scalars = self._get_signal_scalars(info["zip_path"], info["internal_path"])

        start_scalar = info["start_sample"] * 2
        end_scalar = start_scalar + (self.window_size * 2)

        if end_scalar <= len(raw_scalars):
            window_scalars = raw_scalars[start_scalar:end_scalar]
        else:
            window_scalars = raw_scalars[: self.window_size * 2]

        i_samples = window_scalars[0::2]
        q_samples = window_scalars[1::2]
        complex_window = (i_samples + 1j * q_samples).astype(np.complex64)

        # Centralized Preprocessing
        clean_window = self.preprocessor.process(complex_window)

        # Real channels [2, N]
        iq_2ch = complex_to_channels(clean_window)
        tensor_x = torch.from_numpy(iq_2ch).float()
        target_y = info["target"]
        snr_val = info["snr_db"]

        return tensor_x, target_y, snr_val


def get_cspb_dataloader(
    split_df: pd.DataFrame,
    classes: List[str] = CSPB_CLASSES,
    batch_size: int = 64,
    shuffle: bool = True,
    window_size: int = 2048,
    windows_per_signal: int = 4,
    num_workers: int = 0,
    preprocessor: Optional[IQPreprocessor] = None,
    preload_memory: bool = True,
) -> DataLoader:
    dataset = CSPBStreamingDataset(
        split_df=split_df,
        classes=classes,
        window_size=window_size,
        windows_per_signal=windows_per_signal,
        preprocessor=preprocessor,
        preload_memory=preload_memory,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
