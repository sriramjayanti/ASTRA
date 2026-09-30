import os
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ASTRA_DATASET")

# Standard CSPB 9-column schema
CSPB_COLUMNS = [
    "signal_index",
    "modulation",
    "t0",
    "carrier_offset",
    "rolloff",
    "u",
    "d",
    "snr_db",
    "noise_density_db"
]

# Supported binary dtypes for .tim files
DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int16": np.int16,
    "int32": np.int32
}


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Loads configuration YAML file."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def read_tim_file(filepath: Union[str, Path], dtype: str = "float32") -> np.ndarray:
    """
    Reads a CSPB or ASTRA synthetic IQ file (.tim, .iq, .wav) with automatic dtype detection.
    
    Args:
        filepath: Path to the .tim, .iq, or .wav file
        dtype: Fallback data type string ("float32", "float64", "int16", "int32", "int8")
        
    Returns:
        complex_samples: np.ndarray[complex64]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Signal file not found: {filepath}")

    if filepath.suffix.lower() == ".wav":
        from scipy.io import wavfile
        _, data = wavfile.read(str(filepath))
        if data.ndim == 2:
            i_samples = data[:, 0].astype(np.float32)
            q_samples = data[:, 1].astype(np.float32)
        else:
            i_samples = data[0::2].astype(np.float32)
            q_samples = data[1::2].astype(np.float32)
        if np.issubdtype(data.dtype, np.integer):
            max_val = float(np.iinfo(data.dtype).max)
            i_samples = i_samples / max_val
            q_samples = q_samples / max_val
        return (np.nan_to_num(i_samples) + 1j * np.nan_to_num(q_samples)).astype(np.complex64)

    # Auto-detect storage dtype from companion .capture.json if available
    file_dtype = dtype
    cap_json_candidates = [
        filepath.parent.parent / "visible_metadata" / f"{filepath.stem}.capture.json",
        filepath.with_suffix(".capture.json"),
    ]
    for cj in cap_json_candidates:
        if cj.is_file():
            try:
                with open(cj, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "storage_dtype" in meta:
                    file_dtype = meta["storage_dtype"]
                    break
            except Exception:
                pass

    if file_dtype == "int8":
        np_dtype = np.int8
    else:
        np_dtype = DTYPE_MAP.get(file_dtype, np.float32)

    raw = np.fromfile(str(filepath), dtype=np_dtype)

    if len(raw) % 2 != 0:
        raw = raw[: len(raw) - 1]

    if len(raw) == 0:
        return np.zeros(2048, dtype=np.complex64)

    # Interleaved format: I0, Q0, I1, Q1, ...
    i_samples = raw[0::2].astype(np.float32)
    q_samples = raw[1::2].astype(np.float32)

    # If integer, scale to [-1.0, +1.0]
    if np.issubdtype(np_dtype, np.integer):
        max_val = float(np.iinfo(np_dtype).max)
        i_samples = i_samples / max_val
        q_samples = q_samples / max_val

    i_samples = np.nan_to_num(i_samples, nan=0.0, posinf=0.0, neginf=0.0)
    q_samples = np.nan_to_num(q_samples, nan=0.0, posinf=0.0, neginf=0.0)

    complex_samples = i_samples + 1j * q_samples
    return complex_samples.astype(np.complex64)


class IQPreprocessor:
    """
    Centralized preprocessor for complex IQ samples.
    Ensures identical preprocessing between training, validation, evaluation, and inference.
    """
    def __init__(self, remove_dc: bool = True, normalization: str = "rms", epsilon: float = 1e-8):
        self.remove_dc = remove_dc
        self.normalization = normalization.lower().strip()
        self.epsilon = float(epsilon)

    def __call__(self, iq_samples: np.ndarray) -> np.ndarray:
        return self.process(iq_samples)

    def process(self, iq_samples: np.ndarray) -> np.ndarray:
        # 1. Sanitize NaN / Inf
        iq_samples = np.nan_to_num(iq_samples, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. DC Offset Removal
        if self.remove_dc and len(iq_samples) > 0:
            m = np.mean(iq_samples)
            if np.isfinite(m):
                iq_samples = iq_samples - m

        # 3. Amplitude Normalization
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


def preprocess_iq(
    iq_samples: np.ndarray,
    remove_dc: bool = True,
    normalization: str = "rms",
    eps: float = 1e-8
) -> np.ndarray:
    """Functional interface for backwards compatibility."""
    preprocessor = IQPreprocessor(remove_dc=remove_dc, normalization=normalization, epsilon=eps)
    return preprocessor.process(iq_samples)


def complex_to_channels(complex_window: np.ndarray) -> np.ndarray:
    """
    Converts 1D complex IQ window [N] to 2D real tensor [2, N]
    channel 0 = In-phase (I)
    channel 1 = Quadrature (Q)
    """
    i_ch = np.real(complex_window).astype(np.float32)
    q_ch = np.imag(complex_window).astype(np.float32)
    return np.stack([i_ch, q_ch], axis=0)


def load_truth_metadata(truth_path: Union[str, Path]) -> pd.DataFrame:
    """
    Robust parser for CSPB Truth metadata files.
    Handles CSV, TSV, JSON, and headerless whitespace-delimited CSPB truth files.
    """
    truth_path = Path(truth_path)
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    suffix = truth_path.suffix.lower()

    if suffix == ".json":
        df = pd.read_json(truth_path)
    else:
        # Peek at first line to check headers and delimiter
        with open(truth_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()

        # Check if first line contains known column names
        has_header = any(col in first_line.lower() for col in ["signal", "modulation", "snr", "t0", "mod"])

        delimiter = "," if "," in first_line else (r"\s+" if " " in first_line or "\t" in first_line else ",")
        
        if has_header:
            df = pd.read_csv(truth_path, sep=delimiter, engine="python")
        else:
            # Headerless CSPB standard file
            df = pd.read_csv(truth_path, sep=delimiter, header=None, engine="python")
            if len(df.columns) == len(CSPB_COLUMNS):
                df.columns = CSPB_COLUMNS
            elif len(df.columns) >= 2:
                # Assign default names for available columns
                assigned_cols = [CSPB_COLUMNS[i] if i < len(CSPB_COLUMNS) else f"col_{i}" for i in range(len(df.columns))]
                df.columns = assigned_cols

    # Standardize column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    alias_map = {
        "signal": "signal_index",
        "index": "signal_index",
        "file": "signal_index",
        "dataset_record_id": "signal_index",
        "capture_id": "signal_index",
        "capture_path": "resolved_file_path",
        "mod": "modulation",
        "modulation_type": "modulation",
        "class": "modulation",
        "label": "modulation",
        "snr": "snr_db",
        "in_band_snr": "snr_db",
        "t_0": "t0",
        "cfo": "carrier_offset",
        "carrier": "carrier_offset",
        "excess_bandwidth": "rolloff",
        "roll_off": "rolloff",
    }
    for old_col, new_col in alias_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    if "modulation" in df.columns:
        df["modulation"] = df["modulation"].astype(str).str.lower().str.strip()

    return df


def find_signal_file(data_dir: Path, sig_idx: Any) -> Optional[Path]:
    """Helper to locate .tim or .iq file on disk by signal index or path."""
    if sig_idx is not None:
        p_direct = Path(str(sig_idx))
        if p_direct.is_file():
            return p_direct

    file_candidates = [
        data_dir / f"signal_{sig_idx}.tim",
        data_dir / f"signal_{sig_idx:04d}.tim" if isinstance(sig_idx, int) else None,
        data_dir / f"{sig_idx}.tim",
        data_dir / f"{sig_idx}.iq",
        data_dir / f"captures/{sig_idx}.iq",
        data_dir / f"{sig_idx}",
    ]
    for cand in file_candidates:
        if cand is not None and cand.is_file():
            return cand
    return None


def create_file_level_splits(
    data_dir: Union[str, Path],
    truth_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
    stratify_by_mod: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates file-level train/val/test splits ensuring:
    1. Every referenced signal file is verified to exist on disk.
    2. Zero signal file leakage across splits.
    3. Robust allocation guaranteeing at least 1 sample per split when possible.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    data_path = Path(data_dir)
    rng = np.random.default_rng(seed)

    # 1. Verify existence of every file in truth_df
    valid_records = []
    missing_files = []
    for idx, row in truth_df.iterrows():
        file_cand = row.get("resolved_file_path", row.get("capture_path", None))
        found_path = None
        if file_cand is not None and Path(str(file_cand)).is_file():
            found_path = Path(str(file_cand))
        else:
            sig_idx = row.get("signal_index", None)
            found_path = find_signal_file(data_path, sig_idx)
            if found_path is None and "capture_id" in row:
                found_path = find_signal_file(data_path, row["capture_id"])
            if found_path is None and file_cand is not None:
                found_path = find_signal_file(data_path, Path(str(file_cand)).name)

        if found_path is not None:
            row_dict = row.to_dict()
            row_dict["resolved_file_path"] = str(found_path)
            valid_records.append(row_dict)
        else:
            missing_files.append(str(row.get("signal_index", idx)))

    if missing_files:
        logger.warning(
            f"Found {len(missing_files)} signal references in truth metadata with missing files on disk. "
            f"First few missing: {missing_files[:5]}"
        )

    if not valid_records:
        raise FileNotFoundError(f"None of the {len(truth_df)} signal files referenced in truth metadata exist in {data_path}!")

    verified_df = pd.DataFrame(valid_records)
    logger.info(f"Verified {len(verified_df)} / {len(truth_df)} signal files existing on disk in {data_path}")

    # 2. File-level split allocation with robust minimum sample guarantee
    if stratify_by_mod and "modulation" in verified_df.columns:
        train_indices, val_indices, test_indices = [], [], []
        for mod, group in verified_df.groupby("modulation"):
            idx_list = group.index.to_numpy()
            rng.shuffle(idx_list)
            n_total = len(idx_list)
            
            if n_total >= 3:
                n_train = max(1, int(round(n_total * train_ratio)))
                n_val = max(1, int(round(n_total * val_ratio)))
                # Adjust if train+val takes all
                if n_train + n_val >= n_total:
                    n_train = n_total - 2
                    n_val = 1
                n_test = n_total - (n_train + n_val)
            elif n_total == 2:
                n_train, n_val, n_test = 1, 1, 0
            else:
                n_train, n_val, n_test = 1, 0, 0

            train_indices.extend(idx_list[:n_train])
            val_indices.extend(idx_list[n_train:n_train + n_val])
            test_indices.extend(idx_list[n_train + n_val:])
        
        train_df = verified_df.loc[train_indices].reset_index(drop=True)
        val_df = verified_df.loc[val_indices].reset_index(drop=True)
        test_df = verified_df.loc[test_indices].reset_index(drop=True)
    else:
        indices = np.arange(len(verified_df))
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        
        train_df = verified_df.iloc[indices[:n_train]].reset_index(drop=True)
        val_df = verified_df.iloc[indices[n_train:n_train + n_val]].reset_index(drop=True)
        test_df = verified_df.iloc[indices[n_train + n_val:]].reset_index(drop=True)

    # 3. Zero Leakage Assertion
    train_files = set(train_df["resolved_file_path"])
    val_files = set(val_df["resolved_file_path"])
    test_files = set(test_df["resolved_file_path"])
    
    assert len(train_files.intersection(val_files)) == 0, "Leakage detected between Train and Val!"
    assert len(train_files.intersection(test_files)) == 0, "Leakage detected between Train and Test!"
    assert len(val_files.intersection(test_files)) == 0, "Leakage detected between Val and Test!"
    logger.info("Verified ZERO signal file leakage across Train/Val/Test splits.")

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        train_df.to_json(out_path / "train_manifest.json", orient="records", indent=2)
        val_df.to_json(out_path / "val_manifest.json", orient="records", indent=2)
        test_df.to_json(out_path / "test_manifest.json", orient="records", indent=2)
        logger.info(f"Saved split manifests to {out_path}")

    return train_df, val_df, test_df


class IQSignalDataset(Dataset):
    """
    High-Performance PyTorch Dataset for 1D IQ Windows [2, window_size].
    Uses memory-mapped (np.memmap) file access to eliminate redundant I/O on large datasets.
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        data_dir: Union[str, Path],
        tim_dtype: str,
        classes: List[str],
        window_size: int = 2048,
        stride: Optional[int] = None,
        preprocessor: Optional[IQPreprocessor] = None,
        use_memmap: bool = True
    ):
        if tim_dtype not in DTYPE_MAP:
            raise ValueError(f"Invalid tim_dtype '{tim_dtype}'. Supported: {list(DTYPE_MAP.keys())}")

        self.data_dir = Path(data_dir)
        self.tim_dtype = tim_dtype
        self.np_dtype = DTYPE_MAP[tim_dtype]
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.preprocessor = preprocessor or IQPreprocessor()
        self.use_memmap = use_memmap

        self.samples = []
        self._memmaps = {}  # Cache opened memmaps per process/worker

        self._index_dataset(manifest_df)

    def _index_dataset(self, manifest_df: pd.DataFrame):
        """Indexes slice offsets for every window in each signal file."""
        for _, row in manifest_df.iterrows():
            mod_label = str(row.get("modulation", "")).lower().strip()
            if mod_label not in self.class_to_idx:
                continue

            target_idx = self.class_to_idx[mod_label]
            snr_val = float(row.get("snr_db", 0.0)) if "snr_db" in row else 0.0
            sig_idx = row.get("signal_index", None)

            # Locate file path
            file_path = row.get("resolved_file_path", None)
            if file_path is None or not Path(file_path).is_file():
                found = find_signal_file(self.data_dir, sig_idx)
                if found is None:
                    continue
                file_path = str(found)

            # Get sample length from file size without full reading
            file_size_bytes = os.path.getsize(file_path)
            item_size = np.dtype(self.np_dtype).itemsize
            total_scalars = file_size_bytes // item_size
            total_complex_samples = total_scalars // 2

            if total_complex_samples < self.window_size:
                continue

            num_windows = (total_complex_samples - self.window_size) // self.stride + 1
            for w_idx in range(num_windows):
                start_sample = w_idx * self.stride
                self.samples.append({
                    "file_path": file_path,
                    "start_sample": start_sample,
                    "target": target_idx,
                    "modulation": mod_label,
                    "snr_db": snr_val,
                    "signal_index": sig_idx
                })

        logger.info(f"Indexed {len(self.samples)} windows of size {self.window_size} from {len(manifest_df)} files.")

    def _get_memmap(self, file_path: str) -> np.memmap:
        if file_path not in self._memmaps:
            self._memmaps[file_path] = np.memmap(file_path, dtype=self.np_dtype, mode="r")
        return self._memmaps[file_path]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        info = self.samples[idx]
        file_path = info["file_path"]
        start_sample = info["start_sample"]
        complex_window = None

        if self.use_memmap and not file_path.lower().endswith(".wav"):
            try:
                mmap_arr = self._get_memmap(file_path)
                start_scalar = start_sample * 2
                end_scalar = start_scalar + (self.window_size * 2)
                raw_slice = np.array(mmap_arr[start_scalar:end_scalar], copy=True)
                i_samples = raw_slice[0::2].astype(np.float32)
                q_samples = raw_slice[1::2].astype(np.float32)
                complex_window = i_samples + 1j * q_samples
            except Exception:
                complex_window = None

        if complex_window is None or len(complex_window) < self.window_size:
            raw_iq = read_tim_file(file_path, dtype=self.tim_dtype)
            if len(raw_iq) >= start_sample + self.window_size:
                complex_window = raw_iq[start_sample : start_sample + self.window_size]
            elif len(raw_iq) > start_sample:
                sl = raw_iq[start_sample:]
                complex_window = np.pad(sl, (0, self.window_size - len(sl)))
            else:
                complex_window = np.zeros(self.window_size, dtype=np.complex64)

        # Centralized Preprocessing
        clean_window = self.preprocessor.process(complex_window)

        # Convert to [2, N]
        iq_2ch = complex_to_channels(clean_window)
        tensor_x = torch.from_numpy(iq_2ch).float()
        target_y = info["target"]
        snr_val = info["snr_db"]

        return tensor_x, target_y, snr_val


def get_dataloader(
    manifest_df: pd.DataFrame,
    data_dir: Union[str, Path],
    tim_dtype: str,
    classes: List[str],
    batch_size: int = 64,
    shuffle: bool = True,
    window_size: int = 2048,
    stride: Optional[int] = None,
    num_workers: int = 0,
    preprocessor: Optional[IQPreprocessor] = None
) -> DataLoader:
    dataset = IQSignalDataset(
        manifest_df=manifest_df,
        data_dir=data_dir,
        tim_dtype=tim_dtype,
        classes=classes,
        window_size=window_size,
        stride=stride,
        preprocessor=preprocessor,
        use_memmap=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
