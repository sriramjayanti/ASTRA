"""
Production-quality FEC (Forward Error Correction) Generator for ASTRA.
Applies configurable channel coding (None, Convolutional, Reed-Solomon, Concatenated, LDPC)
to FrameRecord bitstreams while preserving exact ground-truth parameters and telemetry.
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

from ..framing.models import FrameRecord
from .models import FECRecord
from .padding import pad_to_multiple
from .convolutional import ConvolutionalCode
from .reed_solomon import ReedSolomonCode
from .concatenated import ConcatenatedCode
from .ldpc import LDPCCode, LDPC_PROFILES
from .profiles import BUILTIN_FEC_PROFILES, get_fec_profile
from .validators import validate_fec_record

logger = logging.getLogger("astra_synthetic.fec")

DEFAULT_FEC_CONFIG: dict[str, Any] = {
    "fec_generator": {
        "version": "1.0.0",
        "master_seed": 42,
        "bit_order": "msb_first",
        "enabled_types": [
            "none",
            "convolutional",
            "reed_solomon",
            "concatenated",
            "ldpc",
        ],
        "selection_probability": {
            "none": 0.10,
            "convolutional": 0.35,
            "reed_solomon": 0.20,
            "concatenated": 0.15,
            "ldpc": 0.20,
        },
        "default_profiles": {
            "none": "none",
            "convolutional": "conv_k7_r12",
            "reed_solomon": "rs_255_223",
            "concatenated": "concat_rs255223_conv_k7",
            "ldpc": "ldpc_n128_k64_r12",
        },
    }
}


class FECGenerator:
    """Forward Error Correction Ground-Truth Generator for ASTRA synthetic pipeline."""

    VERSION = "1.0.0"

    def __init__(self, config: dict[str, Any] | str | Path | None = None):
        """Initialize FECGenerator with configuration dictionary or file path.

        Args:
            config: Config dict, YAML file path, or None (uses DEFAULT_FEC_CONFIG).
        """
        self.raw_config = self._load_config(config)
        fec_cfg = self.raw_config.get("fec_generator", self.raw_config)

        self.version = fec_cfg.get("version", self.VERSION)
        self.master_seed = fec_cfg.get("master_seed", 42)
        self.bit_order = fec_cfg.get("bit_order", "msb_first")
        if self.bit_order != "msb_first":
            raise ValueError(f"Unsupported bit order '{self.bit_order}'. ASTRA requires 'msb_first'.")

        self._rng = np.random.default_rng(self.master_seed)
        self._record_counter = 0

        self.enabled_types = fec_cfg.get("enabled_types", ["none", "convolutional", "reed_solomon", "concatenated", "ldpc"])
        self.selection_prob = fec_cfg.get("selection_probability", {
            "none": 0.10,
            "convolutional": 0.35,
            "reed_solomon": 0.20,
            "concatenated": 0.15,
            "ldpc": 0.20,
        })
        self.default_profiles = fec_cfg.get("default_profiles", {
            "none": "none",
            "convolutional": "conv_k7_r12",
            "reed_solomon": "rs_255_223",
            "concatenated": "concat_rs255223_conv_k7",
            "ldpc": "ldpc_n128_k64_r12",
        })

    def _load_config(self, config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        """Load YAML or dictionary configuration."""
        if config is None:
            return DEFAULT_FEC_CONFIG.copy()
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
                        raise ImportError("PyYAML is required to parse .yaml files.")
                    raise
            return loaded if isinstance(loaded, dict) else {}

    def _next_fec_id(self) -> str:
        """Generate next sequential FEC record ID."""
        self._record_counter += 1
        return f"fec_{self._record_counter:06d}"

    def reset(self, seed: int | None = None) -> None:
        """Reset internal generator state, ID counter, and RNG."""
        if seed is not None:
            self.master_seed = seed
        self._rng = np.random.default_rng(self.master_seed)
        self._record_counter = 0

    def encode(
        self,
        frame_record: FrameRecord,
        fec_type: str | None = None,
        profile_name: str | None = None,
        seed: int | None = None,
        fec_record_id: str | None = None,
        **kwargs: Any,
    ) -> FECRecord:
        """Encode a FrameRecord into an FECRecord.

        Args:
            frame_record: Source FrameRecord instance (immutable ground truth).
            fec_type: FEC family name ('none', 'convolutional', 'reed_solomon', 'concatenated', 'ldpc').
            profile_name: Specific profile name (e.g. 'conv_k7_r12', 'rs_255_223', etc.).
            seed: Seed for random selection if applicable.
            fec_record_id: Explicit ID if overriding sequential counter.
            **kwargs: Type-specific parameter overrides.

        Returns:
            Validated FECRecord.
        """
        if not isinstance(frame_record, FrameRecord):
            raise TypeError(f"Expected FrameRecord, got {type(frame_record).__name__}")

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        fid = fec_record_id if fec_record_id is not None else self._next_fec_id()

        # Resolve FEC type and profile
        if profile_name is not None:
            prof_def = get_fec_profile(profile_name)
            chosen_type = prof_def.fec_type
            chosen_profile = prof_def.name
        elif fec_type is not None:
            chosen_type = str(fec_type).strip().lower()
            chosen_profile = self.default_profiles.get(chosen_type)
        else:
            # Random selection according to distribution
            types = [t for t in self.enabled_types if t in self.selection_prob]
            probs = [self.selection_prob[t] for t in types]
            prob_arr = np.array(probs, dtype=np.float64) / sum(probs)
            chosen_type = str(rng.choice(types, p=prob_arr))
            chosen_profile = self.default_profiles.get(chosen_type)

        input_bits = frame_record.frame_bits.copy()
        orig_bit_len = len(input_bits)

        padding_bits = np.empty(0, dtype=np.uint8)
        padding_length = 0
        block_count = 1
        block_boundaries: list[dict[str, Any]] = []
        intermediates: dict[str, np.ndarray] = {}

        # -------------------------------------------------------------
        # Dispatch Coding Families
        # -------------------------------------------------------------
        if chosen_type == "none":
            encoded_bits = input_bits.copy()
            nominal_rate = 1.0
            effective_rate = 1.0
            params: dict[str, Any] = {"mode": "none"}
            block_boundaries = [{
                "block_index": 0,
                "input_start_bit": 0,
                "original_bit_length": orig_bit_len,
                "encoded_start_bit": 0,
                "encoded_bit_length": orig_bit_len,
            }]

        elif chosen_type == "convolutional":
            prof_def = get_fec_profile(chosen_profile) if chosen_profile else None
            p_cfg = prof_def.config if prof_def else {}
            
            k_val = kwargs.get("constraint_length", p_cfg.get("constraint_length", 7))
            gens = kwargs.get("generators", p_cfg.get("generators", [0o171, 0o133]))
            rate_s = kwargs.get("rate", p_cfg.get("rate", "1/2"))
            term = kwargs.get("termination_mode", p_cfg.get("termination_mode", "zero_tail"))

            conv = ConvolutionalCode(
                constraint_length=k_val,
                generators=gens,
                rate=rate_s,
                termination_mode=term,
            )
            encoded_bits, padding_bits, padding_length, params = conv.encode(input_bits)
            nominal_rate = conv.nominal_rate
            effective_rate = orig_bit_len / len(encoded_bits) if len(encoded_bits) > 0 else 0.0
            block_boundaries = [{
                "block_index": 0,
                "input_start_bit": 0,
                "original_bit_length": orig_bit_len,
                "encoded_start_bit": 0,
                "encoded_bit_length": len(encoded_bits),
                "tail_length": padding_length,
            }]

        elif chosen_type == "reed_solomon":
            prof_def = get_fec_profile(chosen_profile) if chosen_profile else None
            p_cfg = prof_def.config if prof_def else {}

            n_val = kwargs.get("n", p_cfg.get("n", 255))
            k_val = kwargs.get("k", p_cfg.get("k", 223))
            sym_b = kwargs.get("symbol_size_bits", p_cfg.get("symbol_size_bits", 8))

            rs = ReedSolomonCode(n=n_val, k=k_val, symbol_size_bits=sym_b)
            encoded_bits, padding_bits, padding_length, block_boundaries, params = rs.encode(input_bits)
            nominal_rate = rs.nominal_rate
            effective_rate = orig_bit_len / len(encoded_bits) if len(encoded_bits) > 0 else 0.0
            block_count = len(block_boundaries)

        elif chosen_type == "concatenated":
            prof_def = get_fec_profile(chosen_profile) if chosen_profile else None
            p_cfg = prof_def.config if prof_def else {}

            outer_name = kwargs.get("outer", p_cfg.get("outer", "rs_255_223"))
            inner_name = kwargs.get("inner", p_cfg.get("inner", "conv_k7_r12"))

            outer_prof = get_fec_profile(outer_name)
            inner_prof = get_fec_profile(inner_name)

            outer_rs = ReedSolomonCode(**outer_prof.config)
            inner_conv = ConvolutionalCode(**inner_prof.config)

            concat = ConcatenatedCode(outer_rs=outer_rs, inner_conv=inner_conv)
            (
                encoded_bits,
                padding_bits,
                padding_length,
                block_boundaries,
                params,
                intermediates,
            ) = concat.encode(input_bits)
            nominal_rate = concat.nominal_rate
            effective_rate = orig_bit_len / len(encoded_bits) if len(encoded_bits) > 0 else 0.0
            block_count = len(block_boundaries)

        elif chosen_type == "ldpc":
            prof_def = get_fec_profile(chosen_profile) if chosen_profile else None
            p_cfg = prof_def.config if prof_def else {}
            pid = kwargs.get("profile_id", p_cfg.get("profile_id", "ldpc_n128_k64_r12"))

            ldpc = LDPCCode(profile=pid)
            encoded_bits, padding_bits, padding_length, block_boundaries, params = ldpc.encode(input_bits)
            nominal_rate = ldpc.nominal_rate
            effective_rate = orig_bit_len / len(encoded_bits) if len(encoded_bits) > 0 else 0.0
            block_count = len(block_boundaries)

        else:
            raise ValueError(
                f"Unsupported FEC type '{chosen_type}'. Available: none, convolutional, reed_solomon, concatenated, ldpc"
            )

        metadata = {
            "bit_order": self.bit_order,
            "generator_version": self.version,
            "source_frame_id": frame_record.frame_id,
        }

        record = FECRecord(
            fec_record_id=fid,
            frame_id=frame_record.frame_id,
            fec_type=chosen_type,
            fec_profile=chosen_profile,
            input_bits=input_bits,
            encoded_bits=encoded_bits,
            input_bit_length=orig_bit_len,
            encoded_bit_length=len(encoded_bits),
            nominal_code_rate=round(nominal_rate, 6),
            effective_code_rate=round(effective_rate, 6),
            padding_bits=padding_bits,
            padding_length=padding_length,
            block_count=block_count,
            block_boundaries=block_boundaries,
            parameters=params,
            metadata=metadata,
            intermediate_stages=intermediates,
        )

        validate_fec_record(record, original_frame=frame_record)
        logger.info(
            "Generated FEC record %s for frame %s (%s, in=%d, out=%d, rate=%.4f)",
            record.fec_record_id,
            record.frame_id,
            record.fec_type,
            record.input_bit_length,
            record.encoded_bit_length,
            record.effective_code_rate,
        )
        return record

    def iter_encoded(
        self,
        frame_records: Iterable[FrameRecord],
        fec_type: str | None = None,
        profile_name: str | None = None,
        balanced: bool = False,
    ) -> Generator[FECRecord, None, None]:
        """Stream/yield FEC-encoded records one by one.

        Args:
            frame_records: Iterable of input FrameRecord objects.
            fec_type: Optional fixed FEC type override.
            profile_name: Optional fixed profile override.
            balanced: If True, cycles evenly through all enabled FEC families.

        Yields:
            FECRecord objects.
        """
        types_cycle = self.enabled_types.copy() if balanced else None
        idx = 0

        for frame in frame_records:
            if balanced and types_cycle:
                curr_type = types_cycle[idx % len(types_cycle)]
                yield self.encode(frame_record=frame, fec_type=curr_type)
            else:
                yield self.encode(frame_record=frame, fec_type=fec_type, profile_name=profile_name)
            idx += 1

    def encode_batch(
        self,
        frame_records: Sequence[FrameRecord],
        fec_type: str | None = None,
        profile_name: str | None = None,
        balanced: bool = False,
    ) -> list[FECRecord]:
        """Encode a sequence of FrameRecords into a list of FECRecords."""
        return list(
            self.iter_encoded(
                frame_records=frame_records,
                fec_type=fec_type,
                profile_name=profile_name,
                balanced=balanced,
            )
        )
