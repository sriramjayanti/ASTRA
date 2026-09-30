"""
dataset.py
PyTorch Dataset and DataLoader builder for bitstream sequence labeling.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .models import StructureLabel, LABEL_NAMES
from .windowing import slice_sequence_windows
from .utils import generate_annotated_synthetic_stream


class BitstreamStructureDataset(Dataset):
    """
    PyTorch Dataset for multi-channel bitstream structure sequence labeling.
    Each item returns:
      - 'channels': Tensor [C, L]
      - 'padding_mask': Tensor [L] (bool: True = padded)
      - 'labels': Tensor [L] (int64: 0..5 or -100 for ignore)
      - 'boundary_targets': Tensor [L] (float32: 1.0 at label transitions)
      - 'frame_targets': Tensor [4] (float32: multi-label flags)
    """

    def __init__(
        self,
        windows: np.ndarray,
        padding_masks: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Args:
            windows: [N_samples, C, L] (float32)
            padding_masks: [N_samples, L] (bool)
            labels: [N_samples, L] (int64)
            metadata: Optional list of metadata dicts per sample
        """
        self.windows = torch.from_numpy(windows).float()
        self.padding_masks = torch.from_numpy(padding_masks).bool()
        self.labels = torch.from_numpy(labels).long()
        self.metadata = metadata or []

        # Precompute boundary transition targets [N_samples, L]
        self.boundary_targets = self._compute_boundary_targets()

        # Precompute frame-level multi-label targets [N_samples, 4]
        # [valid_structure, header_present, payload_present, crc_present]
        self.frame_targets = self._compute_frame_targets()

    def _compute_boundary_targets(self) -> torch.Tensor:
        """Compute 1.0 at label transition positions."""
        num_samples, seq_len = self.labels.shape
        b_targets = torch.zeros((num_samples, seq_len), dtype=torch.float32)

        for i in range(num_samples):
            lbls = self.labels[i].numpy()
            pads = self.padding_masks[i].numpy()
            valid_len = np.sum(~pads)
            if valid_len > 1:
                valid_lbls = lbls[:valid_len]
                diffs = np.diff(valid_lbls) != 0
                transition_indices = np.where(diffs)[0] + 1
                b_targets[i, transition_indices] = 1.0

        return b_targets

    def _compute_frame_targets(self) -> torch.Tensor:
        """Compute frame presence multi-label flags."""
        num_samples = self.labels.shape[0]
        f_targets = torch.zeros((num_samples, 4), dtype=torch.float32)

        for i in range(num_samples):
            lbls = self.labels[i]
            pads = self.padding_masks[i]
            valid_lbls = lbls[~pads]
            if len(valid_lbls) > 0:
                has_sync = (valid_lbls == StructureLabel.SYNC.value).any().item()
                has_header = (valid_lbls == StructureLabel.HEADER.value).any().item()
                has_payload = (valid_lbls == StructureLabel.PAYLOAD.value).any().item()
                has_crc = (valid_lbls == StructureLabel.CRC.value).any().item()

                f_targets[i, 0] = float(has_sync and has_header) # Valid structure
                f_targets[i, 1] = float(has_header)
                f_targets[i, 2] = float(has_payload)
                f_targets[i, 3] = float(has_crc)

        return f_targets

    def __len__(self) -> int:
        return self.windows.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "channels": self.windows[idx],
            "padding_mask": self.padding_masks[idx],
            "labels": self.labels[idx],
            "boundary_targets": self.boundary_targets[idx],
            "frame_targets": self.frame_targets[idx]
        }

    def compute_class_weights(self, num_classes: int = 6) -> torch.Tensor:
        """
        Calculate balanced inverse-frequency class weights for handling class imbalance.
        """
        valid_labels = self.labels[~self.padding_masks].view(-1).numpy()
        valid_labels = valid_labels[valid_labels >= 0] # exclude ignore_index -100

        counts = np.bincount(valid_labels, minlength=num_classes)
        total = len(valid_labels)

        # Inverse frequency weighting
        weights = total / (float(num_classes) * np.maximum(counts, 1.0))
        # Normalize so mean weight is 1.0
        weights = weights / np.mean(weights)
        return torch.from_numpy(weights).float()


def create_synthetic_dataset(
    num_streams: int = 40,
    window_length: int = 2048,
    overlap_fraction: float = 0.5,
    seed: int = 42
) -> BitstreamStructureDataset:
    """
    Generate synthetic dataset covering diverse frame lengths, headers, CRC types, and BER levels.
    """
    rng = np.random.RandomState(seed)

    frame_lengths = [128, 256, 512, 1024]
    sync_words = ["EB90", "1ACFFC1D", "FAF334", "D391"]
    header_lengths = [16, 24, 32, 48]
    crc_lengths = [0, 8, 16, 32]
    ber_levels = [0.0, 0.0001, 0.001, 0.01]

    all_windows = []
    all_masks = []
    all_labels = []

    for s_idx in range(num_streams):
        fl = rng.choice(frame_lengths)
        sync = rng.choice(sync_words)
        hl = rng.choice(header_lengths)
        crc = rng.choice(crc_lengths)
        ber = rng.choice(ber_levels)
        offset = rng.randint(0, 32)
        n_frames = max(4, int(np.ceil((window_length * 2) / fl)))

        bits, channels, labels, meta = generate_annotated_synthetic_stream(
            frame_length=fl,
            num_frames=n_frames,
            sync_word_hex=sync,
            header_length=hl,
            crc_length=crc,
            prefix_noise_bits=offset,
            ber=ber,
            seed=seed + s_idx
        )

        wins, masks, lbl_wins, _ = slice_sequence_windows(
            channels=channels,
            labels=labels,
            window_length=window_length,
            overlap_fraction=overlap_fraction
        )

        all_windows.append(wins)
        all_masks.append(masks)
        all_labels.append(lbl_wins)

    cat_windows = np.concatenate(all_windows, axis=0)
    cat_masks = np.concatenate(all_masks, axis=0)
    cat_labels = np.concatenate(all_labels, axis=0)

    return BitstreamStructureDataset(
        windows=cat_windows,
        padding_masks=cat_masks,
        labels=cat_labels
    )
