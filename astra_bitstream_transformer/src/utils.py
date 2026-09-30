"""
utils.py
Checkpoint serialization, ground-truth synthetic data generation, and evaluation helpers.
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os
import json
import torch
import numpy as np

from .models import StructureLabel, LABEL_NAMES
from .augmentations import apply_ber_augmentation, apply_offset_augmentation


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """Save full training checkpoint with model weights, optimizer, and metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics or {},
        "config": config or {},
        "model_version": "astra_bitstream_cnn_transformer_v1",
        "input_schema_version": "bitstream_model_input_v1",
        "labels": LABEL_NAMES
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Load model weights and metadata from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert hex string to uint8 array."""
    clean_hex = hex_str.strip().lower().replace("0x", "")
    bits = []
    for c in clean_hex:
        v = int(c, 16)
        for s in (3, 2, 1, 0):
            bits.append((v >> s) & 1)
    return np.array(bits, dtype=np.uint8)


def generate_annotated_synthetic_stream(
    frame_length: int = 512,
    num_frames: int = 16,
    sync_word_hex: str = "EB90",
    header_length: int = 32,
    crc_length: int = 16,
    padding_length: int = 0,
    prefix_noise_bits: int = 0,
    ber: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic bitstream with exact ground-truth per-bit structure annotations.

    Frame Structure:
      [Sync Word][Header][Payload][CRC][Padding]

    Labels:
      0: UNKNOWN (prefix/suffix noise)
      1: SYNC
      2: HEADER
      3: PAYLOAD
      4: CRC
      5: PADDING

    Returns:
        Tuple:
          - raw_bits: [N] (uint8 {0, 1})
          - channels: [6, N] (float32 multi-channel sequence tensor)
          - labels: [N] (int64 structure label IDs)
          - metadata: Ground-truth metadata dict
    """
    rng = np.random.RandomState(seed)
    sync_bits = hex_to_bits(sync_word_hex)
    sync_len = len(sync_bits)

    assert frame_length >= sync_len + header_length + crc_length + padding_length, "Frame length too short."
    payload_len = frame_length - sync_len - header_length - crc_length - padding_length

    all_bits: List[int] = []
    all_labels: List[int] = []

    # 1. Optional prefix noise bits (labeled UNKNOWN = 0)
    if prefix_noise_bits > 0:
        p_noise = rng.randint(0, 2, size=prefix_noise_bits, dtype=np.uint8).tolist()
        all_bits.extend(p_noise)
        all_labels.extend([StructureLabel.UNKNOWN.value] * prefix_noise_bits)

    header_template = rng.randint(0, 2, size=header_length, dtype=np.uint8)

    # 2. Frames
    for f_idx in range(num_frames):
        # Sync
        all_bits.extend(sync_bits.tolist())
        all_labels.extend([StructureLabel.SYNC.value] * sync_len)

        # Header (with incrementing sequence count in first 8 bits)
        hdr = header_template.copy()
        if header_length >= 8:
            hdr[:8] = [(f_idx >> (7 - b)) & 1 for b in range(8)]
        all_bits.extend(hdr.tolist())
        all_labels.extend([StructureLabel.HEADER.value] * header_length)

        # Payload
        payload = rng.randint(0, 2, size=payload_len, dtype=np.uint8).tolist()
        all_bits.extend(payload)
        all_labels.extend([StructureLabel.PAYLOAD.value] * payload_len)

        # CRC
        if crc_length > 0:
            crc_bits = rng.randint(0, 2, size=crc_length, dtype=np.uint8).tolist()
            all_bits.extend(crc_bits)
            all_labels.extend([StructureLabel.CRC.value] * crc_length)

        # Padding
        if padding_length > 0:
            pad_bits = np.zeros(padding_length, dtype=np.uint8).tolist()
            all_bits.extend(pad_bits)
            all_labels.extend([StructureLabel.PADDING.value] * padding_length)

    raw_bits = np.array(all_bits, dtype=np.uint8)
    labels = np.array(all_labels, dtype=np.int64)

    # Apply BER if requested
    if ber > 0.0:
        noisy_bits = apply_ber_augmentation(raw_bits, ber=ber, rng=rng)
    else:
        noisy_bits = raw_bits.copy()

    # 3. Construct 6-channel input sequence tensor [6, N]
    N = len(noisy_bits)
    channels = np.zeros((6, N), dtype=np.float32)

    # Ch 0: Bipolar bits (-1.0 or +1.0)
    channels[0, :] = 2.0 * noisy_bits.astype(np.float32) - 1.0

    # Ch 1: Soft confidence (1.0 default)
    channels[1, :] = 1.0

    # Ch 2: Local window entropy (sliding window approximation)
    win_w = 64
    for i in range(0, N, 32):
        w = noisy_bits[max(0, i - win_w // 2):min(N, i + win_w // 2)]
        p1 = float(np.mean(w))
        p0 = 1.0 - p1
        if p0 > 1e-12 and p1 > 1e-12:
            ent = - (p0 * np.log2(p0) + p1 * np.log2(p1))
        else:
            ent = 0.0
        channels[2, i:min(N, i + 32)] = float(ent)

    # Ch 3: Frame boundary / periodic ramp
    channels[3, :] = (np.arange(N) % frame_length) / float(frame_length)

    # Ch 4: Sync candidate impulse
    for f in range(num_frames):
        pos = prefix_noise_bits + f * frame_length
        if pos < N:
            channels[4, pos:min(N, pos + sync_len)] = 1.0

    # Ch 5: Positional stability map
    pos_stab = np.zeros(frame_length, dtype=np.float32)
    pos_stab[:sync_len] = 1.0
    pos_stab[sync_len:sync_len + header_length] = 0.8
    channels[5, :] = np.tile(pos_stab, int(np.ceil(N / frame_length)))[:N]

    metadata = {
        "frame_length": frame_length,
        "num_frames": num_frames,
        "sync_word_hex": sync_word_hex,
        "sync_length": sync_len,
        "header_length": header_length,
        "payload_length": payload_len,
        "crc_length": crc_length,
        "padding_length": padding_length,
        "prefix_noise_bits": prefix_noise_bits,
        "ber": ber,
        "total_bits": N
    }

    return noisy_bits, channels, labels, metadata
