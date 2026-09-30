"""
Production-quality Frame / Sync / Header / CRC Generator for ASTRA.
Converts source PayloadRecord objects into complete communication frames with ground-truth region tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator, Iterable, Sequence
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from ..payload.models import (
    PayloadRecord,
    bits_to_bytes,
)
from .models import FrameRecord, FrameStreamRecord
from .sync import SyncWordGenerator
from .header import HeaderGenerator
from .crc import compute_crc, get_crc_profile
from .validators import validate_frame_record

logger = logging.getLogger("astra_synthetic.framing")

DEFAULT_FRAME_CONFIG: dict[str, Any] = {
    "frame_generator": {
        "version": "1.0.0",
        "master_seed": 42,
        "bit_order": "msb_first",
        "sync": {
            "mode": "fixed",
            "format": "hex",
            "value": "1ACFFC1D",
        },
        "header": {
            "schema_name": "astra_default_v1",
            "fields": {
                "version": {
                    "bits": 4,
                    "value": 1,
                },
                "frame_type": {
                    "bits": 4,
                    "value": 0,
                },
                "sequence_number": {
                    "bits": 16,
                    "source": "auto_increment",
                },
                "payload_length": {
                    "bits": 24,
                    "source": "payload_length",
                    "unit": "bits",
                },
                "flags": {
                    "bits": 8,
                    "value": 0,
                },
            },
        },
        "crc": {
            "enabled": True,
            "type": "crc16_ccitt",
            "scope": ["header", "payload"],
        },
        "padding": {
            "enabled": False,
            "align_to_bits": 8,
            "value": 0,
        },
        "inter_frame_gap": {
            "enabled": False,
            "length_bits": 16,
            "mode": "zeros",
        },
    }
}


class FrameGenerator:
    """Frame / Sync / Header / CRC Generator for ASTRA communication frames."""

    VERSION = "1.0.0"

    def __init__(self, config: dict[str, Any] | str | Path | None = None):
        """Initialize FrameGenerator with configuration.

        Args:
            config: Config dict, YAML file path, or None (uses DEFAULT_FRAME_CONFIG).
        """
        self.raw_config = self._load_config(config)
        fg_cfg = self.raw_config.get("frame_generator", self.raw_config)

        self.version = fg_cfg.get("version", self.VERSION)
        self.master_seed = fg_cfg.get("master_seed", 42)
        self.bit_order = fg_cfg.get("bit_order", "msb_first")
        if self.bit_order != "msb_first":
            raise ValueError(f"Unsupported bit order '{self.bit_order}'. ASTRA requires 'msb_first'.")

        self._rng = np.random.default_rng(self.master_seed)
        self._frame_counter = 0
        self._sequence_number = 0

        # Sub-modules
        sync_cfg = fg_cfg.get("sync", {})
        self.sync_generator = SyncWordGenerator(config=sync_cfg, rng=self._rng)

        header_cfg = fg_cfg.get("header", {})
        self.header_generator = HeaderGenerator(config=header_cfg)

        self.crc_cfg = fg_cfg.get("crc", {"enabled": True, "type": "crc16_ccitt", "scope": ["header", "payload"]})
        self.padding_cfg = fg_cfg.get("padding", {"enabled": False})
        self.gap_cfg = fg_cfg.get("inter_frame_gap", {"enabled": False})

    def _load_config(self, config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        """Load YAML or dictionary configuration."""
        if config is None:
            return DEFAULT_FRAME_CONFIG.copy()
        if isinstance(config, dict):
            return config
        
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if HAS_YAML and (path.suffix.lower() in [".yaml", ".yml"]):
                loaded = yaml.safe_load(f)
            else:
                import json
                try:
                    loaded = json.load(f)
                except Exception:
                    if not HAS_YAML:
                        raise ImportError("PyYAML is required to parse .yaml files. Please install pyyaml.")
                    raise
            return loaded if isinstance(loaded, dict) else {}

    def _next_frame_id(self) -> str:
        """Generate next sequential frame ID."""
        self._frame_counter += 1
        return f"frame_{self._frame_counter:06d}"

    def reset(self, seed: int | None = None) -> None:
        """Reset sequence number, frame counter, and RNG state."""
        if seed is not None:
            self.master_seed = seed
        self._rng = np.random.default_rng(self.master_seed)
        self.sync_generator.set_rng(self._rng)
        self._frame_counter = 0
        self._sequence_number = 0

    def generate(
        self,
        payload_record: PayloadRecord,
        sequence_number: int | None = None,
        frame_id: str | None = None,
        custom_header_fields: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> FrameRecord:
        """Construct a complete communication frame from a PayloadRecord.

        Layout: | SYNC | HEADER | PAYLOAD | CRC |

        Args:
            payload_record: Source PayloadRecord.
            sequence_number: Explicit sequence number (if None, auto-increments).
            frame_id: Explicit frame ID (if None, auto-increments).
            custom_header_fields: Optional overrides for header fields.
            **kwargs: Additional generation options.

        Returns:
            Validated FrameRecord with exact ground-truth boundary tracking.
        """
        if not isinstance(payload_record, PayloadRecord):
            raise TypeError(f"Expected PayloadRecord, got {type(payload_record).__name__}")

        fid = frame_id if frame_id is not None else self._next_frame_id()

        # Resolve sequence number
        if sequence_number is not None:
            seq_num = sequence_number
        else:
            seq_num = self._sequence_number
            self._sequence_number += 1

        # 1. Payload bits (handling optional padding if configured)
        raw_payload_bits = payload_record.payload_bits
        padded_payload_bits = raw_payload_bits
        padding_info: dict[str, Any] = {"enabled": False}

        if self.padding_cfg.get("enabled", False):
            align_bits = int(self.padding_cfg.get("align_to_bits", 8))
            pad_val = int(self.padding_cfg.get("value", 0))
            rem = len(raw_payload_bits) % align_bits
            if rem != 0:
                pad_len = align_bits - rem
                pad_arr = np.full(pad_len, pad_val, dtype=np.uint8)
                padded_payload_bits = np.concatenate([raw_payload_bits, pad_arr])
                padding_info = {
                    "enabled": True,
                    "align_to_bits": align_bits,
                    "pad_value": pad_val,
                    "original_length": len(raw_payload_bits),
                    "padded_length": len(padded_payload_bits),
                    "padding_length": pad_len,
                }

        # 2. Generate Sync Word
        sync_bits, sync_val_repr, sync_len, sync_mode = self.sync_generator.generate(
            mode=kwargs.get("sync_mode"),
            value=kwargs.get("sync_value"),
            length_bits=kwargs.get("sync_length_bits"),
            seed=kwargs.get("sync_seed"),
        )

        # 3. Generate Header
        header_bits, resolved_fields, field_offsets = self.header_generator.build_header(
            payload_record=payload_record,
            sequence_number=seq_num,
            custom_field_values=custom_header_fields,
        )
        header_len = len(header_bits)

        # 4. Generate CRC
        crc_enabled = self.crc_cfg.get("enabled", True)
        crc_type = self.crc_cfg.get("type", "crc16_ccitt")
        crc_scope = self.crc_cfg.get("scope", ["header", "payload"])

        if crc_enabled:
            crc_scope_chunks: list[np.ndarray] = []
            for reg in crc_scope:
                if reg == "header":
                    crc_scope_chunks.append(header_bits)
                elif reg == "payload":
                    crc_scope_chunks.append(padded_payload_bits)
                elif reg == "sync":
                    crc_scope_chunks.append(sync_bits)
                else:
                    raise ValueError(f"Unknown CRC scope region: '{reg}'")

            crc_input = np.concatenate(crc_scope_chunks) if crc_scope_chunks else np.empty(0, dtype=np.uint8)
            crc_val, crc_bits = compute_crc(crc_input, crc_type=crc_type)
            crc_len = len(crc_bits)
        else:
            crc_val = 0
            crc_bits = np.empty(0, dtype=np.uint8)
            crc_len = 0

        # 5. Assemble complete frame: [SYNC | HEADER | PAYLOAD | CRC]
        frame_bits = np.concatenate([sync_bits, header_bits, padded_payload_bits, crc_bits])
        frame_bit_length = len(frame_bits)
        frame_bytes = bits_to_bytes(frame_bits, padding=True)

        # 6. Calculate Region Offsets
        sync_start = 0
        header_start = sync_len
        payload_start = header_start + header_len
        payload_len = len(padded_payload_bits)
        crc_start = payload_start + payload_len

        # 7. Build Metadata
        metadata: dict[str, Any] = {
            "bit_order": self.bit_order,
            "generator_version": self.version,
            "padding": padding_info,
        }

        frame_record = FrameRecord(
            frame_id=fid,
            payload_id=payload_record.payload_id,
            frame_bits=frame_bits,
            frame_bytes=frame_bytes,
            frame_bit_length=frame_bit_length,
            sync_bits=sync_bits,
            header_bits=header_bits,
            payload_bits=padded_payload_bits,
            crc_bits=crc_bits,
            sync_start=sync_start,
            sync_length=sync_len,
            header_start=header_start,
            header_length=header_len,
            payload_start=payload_start,
            payload_length=payload_len,
            crc_start=crc_start,
            crc_length=crc_len,
            sync_type=sync_mode,
            sync_value=sync_val_repr,
            header_schema=self.header_generator.schema_name,
            header_fields=resolved_fields,
            header_field_offsets=field_offsets,
            crc_type=crc_type if crc_enabled else "none",
            crc_value=f"0x{crc_val:X}" if isinstance(crc_val, int) else str(crc_val),
            crc_scope=crc_scope if crc_enabled else [],
            frame_sequence_number=seq_num,
            metadata=metadata,
        )

        validate_frame_record(frame_record, original_payload=payload_record)
        logger.info("Generated frame %s (seq=%d, bits=%d)", frame_record.frame_id, seq_num, frame_bit_length)
        return frame_record

    def iter_frames(
        self,
        payload_records: Iterable[PayloadRecord],
        start_sequence: int = 0,
    ) -> Generator[FrameRecord, None, None]:
        """Stream/yield generated frames one by one.

        Args:
            payload_records: Iterable of PayloadRecord objects.
            start_sequence: Initial sequence number.

        Yields:
            FrameRecord objects.
        """
        current_seq = start_sequence
        for prec in payload_records:
            yield self.generate(payload_record=prec, sequence_number=current_seq)
            current_seq += 1

    def generate_batch(
        self,
        payload_records: Sequence[PayloadRecord],
        start_sequence: int = 0,
    ) -> list[FrameRecord]:
        """Generate a batch of frames into a list."""
        return list(self.iter_frames(payload_records=payload_records, start_sequence=start_sequence))

    def generate_stream(
        self,
        payload_records: Sequence[PayloadRecord],
        inter_frame_gap: dict[str, Any] | None = None,
        start_sequence: int = 0,
    ) -> FrameStreamRecord:
        """Concatenate multiple frames into a single continuous bitstream for correlation analysis.

        Args:
            payload_records: Sequence of input PayloadRecords.
            inter_frame_gap: Optional gap configuration override (e.g. {'enabled': True, 'length_bits': 16, 'mode': 'zeros'}).
            start_sequence: Initial sequence number.

        Returns:
            FrameStreamRecord containing continuous stream bits and ground-truth frame indexing.
        """
        gap_cfg = inter_frame_gap if inter_frame_gap is not None else self.gap_cfg
        gap_enabled = gap_cfg.get("enabled", False)
        gap_len = int(gap_cfg.get("length_bits", 16)) if gap_enabled else 0
        gap_mode = gap_cfg.get("mode", "zeros")

        stream_chunks: list[np.ndarray] = []
        frame_ids: list[str] = []
        frame_starts: list[int] = []
        frame_lengths: list[int] = []
        sync_positions: list[dict[str, int]] = []
        header_positions: list[dict[str, int]] = []
        payload_positions: list[dict[str, int]] = []
        crc_positions: list[dict[str, int]] = []
        gap_positions: list[dict[str, int]] = []

        current_offset = 0

        for idx, prec in enumerate(payload_records):
            frame = self.generate(payload_record=prec, sequence_number=start_sequence + idx)
            
            frame_ids.append(frame.frame_id)
            frame_starts.append(current_offset)
            frame_lengths.append(frame.frame_bit_length)

            sync_positions.append({
                "start": current_offset + frame.sync_start,
                "length": frame.sync_length,
                "value": frame.sync_value,
            })
            header_positions.append({
                "start": current_offset + frame.header_start,
                "length": frame.header_length,
            })
            payload_positions.append({
                "start": current_offset + frame.payload_start,
                "length": frame.payload_length,
            })
            crc_positions.append({
                "start": current_offset + frame.crc_start,
                "length": frame.crc_length,
            })

            stream_chunks.append(frame.frame_bits)
            current_offset += frame.frame_bit_length

            # Insert inter-frame gap between frames (except after the final frame)
            if gap_enabled and idx < len(payload_records) - 1:
                if gap_mode == "random":
                    gap_bits = self._rng.integers(0, 2, size=gap_len, dtype=np.uint8)
                else:
                    gap_bits = np.zeros(gap_len, dtype=np.uint8)

                gap_positions.append({
                    "start": current_offset,
                    "length": gap_len,
                    "mode": gap_mode,
                })
                stream_chunks.append(gap_bits)
                current_offset += gap_len

        all_bits = np.concatenate(stream_chunks) if stream_chunks else np.empty(0, dtype=np.uint8)
        stream_bytes = bits_to_bytes(all_bits, padding=True)

        return FrameStreamRecord(
            stream_bits=all_bits,
            stream_bytes=stream_bytes,
            total_bit_length=len(all_bits),
            frame_ids=frame_ids,
            frame_start_positions=frame_starts,
            frame_lengths=frame_lengths,
            sync_positions=sync_positions,
            header_positions=header_positions,
            payload_positions=payload_positions,
            crc_positions=crc_positions,
            gap_positions=gap_positions,
            metadata={
                "total_frames": len(frame_ids),
                "inter_frame_gap": gap_cfg,
                "bit_order": self.bit_order,
                "generator_version": self.version,
            },
        )
