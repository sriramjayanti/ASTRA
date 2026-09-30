"""
Main Capture / File Generator for ASTRA (Engine 7).
Converts ChannelRecord impaired IQ into physical raw .iq, .wav, and SigMF capture files
with complete parameter ground truth and separated metadata sidecars.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Generator, Iterable
import numpy as np
import yaml

from astra_synthetic.channel.models import ChannelRecord
from .models import CaptureRecord
from .raw_iq import write_raw_iq_file
from .wav_iq import write_wav_iq_file
from .sigmf_writer import write_sigmf_dataset
from .quantization import apply_clipping
from .serializers import save_capture_metadata_pair
from .validators import validate_capture_record

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "capture_config.yaml"


class CaptureGenerator:
    """Orchestrates writing channel-impaired IQ into standardized SDR file formats."""

    def __init__(self, config: dict[str, Any] | Path | str | None = None):
        if config is None:
            if DEFAULT_CONFIG_PATH.exists():
                with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self.config = {}
        elif isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = config

        gen_cfg = self.config.get("capture_generator", {})
        self.version = gen_cfg.get("version", "1.0.0")
        self.master_seed = gen_cfg.get("master_seed", 42)
        self.output_root = Path(gen_cfg.get("output_root", "output/captures"))
        self.profiles = self.config.get("profiles", {})
        self._counter = 0

    def write(
        self,
        channel_record: ChannelRecord,
        profile_name: str | None = None,
        capture_format: str | None = None,
        output_dir: Path | str | None = None,
        filename: str | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> CaptureRecord:
        """Write a single ChannelRecord into a physical capture file and metadata sidecars.

        Args:
            channel_record: Source ChannelRecord (immutable input).
            profile_name: Named preset profile (e.g. 'raw_f32_le_iq', 'wav_i16_iq').
            capture_format: Explicit container format ('raw_iq', 'wav', 'sigmf').
            output_dir: Destination folder (defaults to configured output_root).
            filename: Explicit file name (or auto-generated).
            seed: Deterministic seed for randomized settings / metadata masking.
            overrides: Optional dictionary of parameter overrides.

        Returns:
            CaptureRecord instance.
        """
        self._counter += 1
        cap_id = f"cap_{self._counter:06d}"

        if seed is None:
            seed = (self.master_seed + self._counter * 10007) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        # 1. Resolve Profile & Parameters
        profile = {}
        if profile_name and profile_name in self.profiles:
            profile = dict(self.profiles[profile_name])
        elif profile_name:
            logger.warning("Profile '%s' not found; using defaults.", profile_name)

        params = dict(profile)
        if overrides:
            params.update(overrides)

        fmt = capture_format or params.get("format", "raw_iq").lower()
        dtype = params.get("dtype", "float32").lower()
        endianness = params.get("endianness", "little").lower()
        iq_order = params.get("iq_order", "IQ").upper()
        center_freq = params.get("center_frequency_hz")

        # Quantization settings
        q_cfg = params.get("quantization", {})
        q_mode = q_cfg.get("mode", "full_scale_peak" if "int" in dtype else "none")
        headroom_db = float(q_cfg.get("headroom_db", 1.0))
        fixed_scale = q_cfg.get("fixed_scale")

        # Clipping settings
        clip_cfg = params.get("clipping", {})
        clipping_enabled = clip_cfg.get("enabled", False)
        clip_level = float(clip_cfg.get("clip_level", 1.0))

        # Metadata masking settings (sample rate exposure)
        mask_cfg = params.get("visible_metadata", {})
        sr_mask_cfg = mask_cfg.get("sample_rate", {})
        expose_prob = float(sr_mask_cfg.get("expose_probability", 1.0))
        if "sample_rate_visible" in params:
            sample_rate_visible = bool(params["sample_rate_visible"])
        else:
            sample_rate_visible = bool(rng.uniform(0.0, 1.0) <= expose_prob)

        # 2. Extract and prepare baseband IQ
        input_iq = np.asarray(channel_record.impaired_iq, dtype=np.complex64)
        Fs = float(channel_record.sample_rate)
        N_complex = len(input_iq)
        N_scalar = 2 * N_complex
        duration_sec = float(N_complex / Fs) if Fs > 0 else 0.0

        source_sha256 = hashlib.sha256(input_iq.tobytes()).hexdigest() if N_complex > 0 else hashlib.sha256(b"").hexdigest()

        current_iq = input_iq.copy()
        clipped_count = 0
        if clipping_enabled:
            current_iq, clipped_count, _ = apply_clipping(current_iq, clip_level=clip_level)

        # 3. Setup Target File Path
        target_dir = Path(output_dir) if output_dir else self.output_root
        target_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            file_name = filename
        else:
            ext = ".iq" if fmt == "raw_iq" else (".wav" if fmt == "wav" else ".sigmf-data")
            file_name = f"{cap_id}{ext}"

        file_path = target_dir / file_name

        # 4. Write File based on Container Format
        if fmt == "raw_iq":
            out_p, file_sha, file_sz, scale, mse, qsnr = write_raw_iq_file(
                iq=current_iq,
                output_path=file_path,
                storage_dtype=dtype,
                endianness=endianness,
                iq_order=iq_order,
                quantization_mode=q_mode,
                headroom_db=headroom_db,
                fixed_scale=fixed_scale,
            )
            q_enabled = "int" in dtype
            q_bits = 16 if "16" in dtype else (8 if "8" in dtype else None)
        elif fmt == "wav":
            out_p, file_sha, file_sz, scale, mse, qsnr = write_wav_iq_file(
                iq=current_iq,
                output_path=file_path,
                sample_rate_hz=Fs,
                storage_dtype=dtype if dtype in ("int16", "float32") else "int16",
                iq_order=iq_order,
                quantization_mode=q_mode,
                headroom_db=headroom_db,
                fixed_scale=fixed_scale,
            )
            q_enabled = "int" in dtype
            q_bits = 16 if "int" in dtype else None
        elif fmt == "sigmf":
            out_p, meta_p, file_sha, file_sz, scale, mse, qsnr = write_sigmf_dataset(
                iq=current_iq,
                output_base_path=file_path,
                sample_rate_hz=Fs,
                center_frequency_hz=center_freq,
                storage_dtype=dtype,
                endianness=endianness,
                quantization_mode=q_mode,
                headroom_db=headroom_db,
            )
            q_enabled = "int" in dtype
            q_bits = 16 if "16" in dtype else (8 if "8" in dtype else None)
        else:
            raise ValueError(f"Unsupported capture format: '{fmt}'")

        metadata = {
            "generator": "ASTRA Capture Generator",
            "version": self.version,
            "seed": seed,
            "profile_name": profile_name,
            "channel_record_id": channel_record.channel_record_id,
            "modulation_record_id": channel_record.modulation_record_id,
            "interleaver_record_id": channel_record.interleaver_record_id,
            "fec_record_id": channel_record.fec_record_id,
            "frame_id": channel_record.frame_id,
        }

        record = CaptureRecord(
            capture_record_id=cap_id,
            channel_record_id=channel_record.channel_record_id,
            modulation_record_id=channel_record.modulation_record_id,
            interleaver_record_id=channel_record.interleaver_record_id,
            fec_record_id=channel_record.fec_record_id,
            frame_id=channel_record.frame_id,
            file_path=str(out_p),
            file_format=fmt,
            sample_rate_hz=Fs,
            center_frequency_hz=center_freq,
            sample_count_complex=N_complex,
            scalar_sample_count=N_scalar,
            duration_seconds=duration_sec,
            storage_dtype=dtype,
            endianness=endianness,
            iq_order=iq_order,
            channel_count=2,
            quantization_enabled=q_enabled,
            quantization_bits=q_bits,
            quantization_scale=scale if q_enabled else None,
            quantization_mse=mse if q_enabled else None,
            quantization_snr_db=qsnr if q_enabled else None,
            clipping_enabled=clipping_enabled,
            clipped_sample_count=clipped_count,
            source_iq_sha256=source_sha256,
            file_sha256=file_sha,
            file_size_bytes=file_sz,
            sample_rate_visible=sample_rate_visible,
            parameters=params,
            metadata=metadata,
        )

        # 5. Write paired metadata sidecars
        save_capture_metadata_pair(record, output_dir=target_dir)

        # 6. Validate record and file integrity
        validate_capture_record(record, original_channel=channel_record)

        return record

    def write_batch(
        self,
        channel_records: list[ChannelRecord],
        profile_name: str | None = None,
        capture_format: str | None = None,
        output_dir: Path | str | None = None,
        base_seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> list[CaptureRecord]:
        """Write a batch of ChannelRecords to capture files."""
        return list(
            self.iter_write(
                channel_records,
                profile_name=profile_name,
                capture_format=capture_format,
                output_dir=output_dir,
                base_seed=base_seed,
                overrides=overrides,
            )
        )

    def iter_write(
        self,
        channel_records: Iterable[ChannelRecord],
        profile_name: str | None = None,
        capture_format: str | None = None,
        output_dir: Path | str | None = None,
        base_seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Generator[CaptureRecord, None, None]:
        """Stream and write capture records incrementally."""
        seed = base_seed if base_seed is not None else self.master_seed
        for idx, chan_rec in enumerate(channel_records):
            item_seed = (seed + idx * 7919) & 0xFFFFFFFF
            yield self.write(
                chan_rec,
                profile_name=profile_name,
                capture_format=capture_format,
                output_dir=output_dir,
                seed=item_seed,
                overrides=overrides,
            )
