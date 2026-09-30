"""
Production-quality Interleaver / Bit-Reordering Generator for ASTRA.
Applies configurable bit permutations (None, Block, Convolutional, Diagonal, Pseudo-Random)
to FECRecord bitstreams while recording exact ground-truth permutations and block telemetry.
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

from ..fec.models import FECRecord
from .models import InterleaverRecord
from .block import BlockInterleaver
from .convolutional import ConvolutionalInterleaver
from .diagonal import DiagonalInterleaver
from .pseudo_random import PseudoRandomInterleaver
from .validators import validate_interleaver_record

logger = logging.getLogger("astra_synthetic.interleaving")


BUILTIN_INTERLEAVER_PROFILES: dict[str, dict[str, Any]] = {
    # 1. No Interleaving
    "none": {
        "type": "none",
        "description": "No interleaving / Identity mapping",
        "params": {},
    },

    # 2. Block Interleavers
    "block_8x8": {
        "type": "block",
        "description": "Block Interleaver 8x8 (64 bits)",
        "params": {"rows": 8, "columns": 8, "write_order": "row_major", "read_order": "column_major"},
    },
    "block_16x16": {
        "type": "block",
        "description": "Block Interleaver 16x16 (256 bits)",
        "params": {"rows": 16, "columns": 16, "write_order": "row_major", "read_order": "column_major"},
    },
    "block_16x32": {
        "type": "block",
        "description": "Rectangular Block Interleaver 16x32 (512 bits)",
        "params": {"rows": 16, "columns": 32, "write_order": "row_major", "read_order": "column_major"},
    },
    "block_32x32": {
        "type": "block",
        "description": "Block Interleaver 32x32 (1024 bits)",
        "params": {"rows": 32, "columns": 32, "write_order": "row_major", "read_order": "column_major"},
    },

    # 3. Convolutional Interleavers
    "conv_b4_d2": {
        "type": "convolutional",
        "description": "Convolutional Interleaver (B=4 branches, M=2 delay step)",
        "params": {"num_branches": 4, "delay_step": 2},
    },
    "conv_b8_d4": {
        "type": "convolutional",
        "description": "Convolutional Interleaver (B=8 branches, M=4 delay step)",
        "params": {"num_branches": 8, "delay_step": 4},
    },

    # 4. Diagonal Interleavers
    "diag_8x8": {
        "type": "diagonal",
        "description": "Diagonal Matrix Interleaver 8x8 (64 bits, TL-BR)",
        "params": {"rows": 8, "columns": 8, "direction": "top_left_to_bottom_right"},
    },
    "diag_16x16": {
        "type": "diagonal",
        "description": "Diagonal Matrix Interleaver 16x16 (256 bits, TL-BR)",
        "params": {"rows": 16, "columns": 16, "direction": "top_left_to_bottom_right"},
    },

    # 5. Pseudo-Random Interleavers
    "pr_256": {
        "type": "pseudo_random",
        "description": "Pseudo-Random Permutation (256 bits)",
        "params": {"block_size": 256, "seed": 42},
    },
    "pr_512": {
        "type": "pseudo_random",
        "description": "Pseudo-Random Permutation (512 bits)",
        "params": {"block_size": 512, "seed": 42},
    },
    "pr_1024": {
        "type": "pseudo_random",
        "description": "Pseudo-Random Permutation (1024 bits)",
        "params": {"block_size": 1024, "seed": 42},
    },
}


DEFAULT_INTERLEAVER_CONFIG: dict[str, Any] = {
    "interleaver_generator": {
        "version": "1.0.0",
        "master_seed": 42,
        "bit_order": "msb_first",
        "enabled_types": [
            "none",
            "block",
            "convolutional",
            "diagonal",
            "pseudo_random",
        ],
        "selection_probability": {
            "none": 0.10,
            "block": 0.30,
            "convolutional": 0.25,
            "diagonal": 0.15,
            "pseudo_random": 0.20,
        },
        "default_profiles": {
            "none": "none",
            "block": "block_16x32",
            "convolutional": "conv_b4_d2",
            "diagonal": "diag_8x8",
            "pseudo_random": "pr_256",
        },
    }
}


class InterleaverGenerator:
    """Configurable Interleaver Generator for ASTRA synthetic pipeline."""

    VERSION = "1.0.0"

    def __init__(
        self,
        config: dict[str, Any] | str | Path | None = None,
        seed: int | None = None,
    ):
        """Initialize InterleaverGenerator.

        Args:
            config: Config dict, YAML file path, or None (uses DEFAULT_INTERLEAVER_CONFIG).
            seed: Optional master seed override.
        """
        self.raw_config = self._load_config(config)
        ig_cfg = self.raw_config.get("interleaver_generator", self.raw_config)

        self.version = ig_cfg.get("version", self.VERSION)
        self.master_seed = seed if seed is not None else ig_cfg.get("master_seed", 42)
        self.bit_order = ig_cfg.get("bit_order", "msb_first")
        if self.bit_order != "msb_first":
            raise ValueError(f"Unsupported bit order '{self.bit_order}'. ASTRA requires 'msb_first'.")

        self.profiles = BUILTIN_INTERLEAVER_PROFILES.copy()
        self._rng = np.random.default_rng(self.master_seed)
        self._record_counter = 0

        self.enabled_types = ig_cfg.get(
            "enabled_types", ["none", "block", "convolutional", "diagonal", "pseudo_random"]
        )
        self.selection_prob = ig_cfg.get("selection_probability", {
            "none": 0.10,
            "block": 0.30,
            "convolutional": 0.25,
            "diagonal": 0.15,
            "pseudo_random": 0.20,
        })
        self.default_profiles = ig_cfg.get("default_profiles", {
            "none": "none",
            "block": "block_16x32",
            "convolutional": "conv_b4_d2",
            "diagonal": "diag_8x8",
            "pseudo_random": "pr_256",
        })

    def _load_config(self, config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        """Load YAML or dictionary configuration."""
        if config is None:
            return DEFAULT_INTERLEAVER_CONFIG.copy()
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

    def _next_interleaver_id(self) -> str:
        """Generate next sequential record ID."""
        self._record_counter += 1
        return f"int_{self._record_counter:06d}"

    def reset(self, seed: int | None = None) -> None:
        """Reset internal generator counter and RNG state."""
        if seed is not None:
            self.master_seed = seed
        self._rng = np.random.default_rng(self.master_seed)
        self._record_counter = 0

    def interleave(
        self,
        fec_record: FECRecord,
        interleaver_type: str | None = None,
        profile_name: str | None = None,
        seed: int | None = None,
        interleaver_record_id: str | None = None,
        **kwargs: Any,
    ) -> InterleaverRecord:
        """Interleave an FECRecord into an InterleaverRecord.

        Args:
            fec_record: Source FECRecord instance (immutable ground truth).
            interleaver_type: Family name ('none', 'block', 'convolutional', 'diagonal', 'pseudo_random').
            profile_name: Name of built-in profile (e.g. 'block_16x32', 'conv_b4_d2', 'pr_256').
            seed: Seed for random selection/padding.
            interleaver_record_id: Explicit record ID override.
            **kwargs: Family-specific parameter overrides.

        Returns:
            Validated InterleaverRecord.
        """
        if not isinstance(fec_record, FECRecord):
            raise TypeError(f"Expected FECRecord, got {type(fec_record).__name__}")

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        iid = interleaver_record_id if interleaver_record_id is not None else self._next_interleaver_id()

        # Resolve type and profile
        if profile_name is not None:
            pname = str(profile_name).strip().lower()
            if pname not in BUILTIN_INTERLEAVER_PROFILES:
                avail = ", ".join(BUILTIN_INTERLEAVER_PROFILES.keys())
                raise ValueError(f"Unknown interleaver profile '{profile_name}'. Available: {avail}")
            pdef = BUILTIN_INTERLEAVER_PROFILES[pname]
            chosen_type = pdef["type"]
            chosen_profile = pname
            prof_params = pdef["params"]
        elif interleaver_type is not None:
            chosen_type = str(interleaver_type).strip().lower()
            chosen_profile = self.default_profiles.get(chosen_type)
            prof_params = BUILTIN_INTERLEAVER_PROFILES.get(chosen_profile, {}).get("params", {})
        else:
            types = [t for t in self.enabled_types if t in self.selection_prob]
            probs = [self.selection_prob[t] for t in types]
            prob_arr = np.array(probs, dtype=np.float64) / sum(probs)
            chosen_type = str(rng.choice(types, p=prob_arr))
            chosen_profile = self.default_profiles.get(chosen_type)
            prof_params = BUILTIN_INTERLEAVER_PROFILES.get(chosen_profile, {}).get("params", {})

        input_bits = fec_record.encoded_bits.copy()
        orig_bit_len = len(input_bits)

        padding_bits = np.empty(0, dtype=np.uint8)
        padding_length = 0
        block_boundaries: list[dict[str, Any]] = []
        perm_array: np.ndarray | None = None
        inv_perm_array: np.ndarray | None = None

        # -------------------------------------------------------------
        # Dispatch Interleaver Families
        # -------------------------------------------------------------
        if chosen_type == "none":
            interleaved_bits = input_bits.copy()
            params = {"mode": "none"}
            block_boundaries = [{
                "block_id": 0,
                "input_start": 0,
                "input_length": orig_bit_len,
                "output_start": 0,
                "output_length": orig_bit_len,
            }]

        elif chosen_type == "block":
            r_val = kwargs.get("rows", prof_params.get("rows", 16))
            c_val = kwargs.get("columns", prof_params.get("columns", 32))
            w_ord = kwargs.get("write_order", prof_params.get("write_order", "row_major"))
            r_ord = kwargs.get("read_order", prof_params.get("read_order", "column_major"))
            pad_m = kwargs.get("pad_mode", prof_params.get("pad_mode", "zeros"))

            interleaver = BlockInterleaver(
                rows=r_val,
                columns=c_val,
                write_order=w_ord,
                read_order=r_ord,
                pad_mode=pad_m,
            )
            interleaved_bits, padding_bits, padding_length, block_boundaries, params = interleaver.interleave(
                bits=input_bits,
                seed=seed,
            )
            perm_array = interleaver.permutation
            inv_perm_array = interleaver.inverse_permutation

        elif chosen_type == "convolutional":
            b_val = kwargs.get("num_branches", prof_params.get("num_branches", 4))
            m_val = kwargs.get("delay_step", prof_params.get("delay_step", 2))

            interleaver = ConvolutionalInterleaver(
                num_branches=b_val,
                delay_step=m_val,
            )
            interleaved_bits, padding_bits, padding_length, block_boundaries, params = interleaver.interleave(
                bits=input_bits,
            )

        elif chosen_type == "diagonal":
            r_val = kwargs.get("rows", prof_params.get("rows", 8))
            c_val = kwargs.get("columns", prof_params.get("columns", 8))
            dir_val = kwargs.get("direction", prof_params.get("direction", "top_left_to_bottom_right"))
            pad_m = kwargs.get("pad_mode", prof_params.get("pad_mode", "zeros"))

            interleaver = DiagonalInterleaver(
                rows=r_val,
                columns=c_val,
                direction=dir_val,
                pad_mode=pad_m,
            )
            interleaved_bits, padding_bits, padding_length, block_boundaries, params = interleaver.interleave(
                bits=input_bits,
                seed=seed,
            )
            perm_array = interleaver.permutation
            inv_perm_array = interleaver.inverse_permutation

        elif chosen_type == "pseudo_random":
            bsize = kwargs.get("block_size", prof_params.get("block_size", 256))
            s_val = kwargs.get("seed", prof_params.get("seed", 42 if seed is None else seed))
            pad_m = kwargs.get("pad_mode", prof_params.get("pad_mode", "zeros"))

            interleaver = PseudoRandomInterleaver(
                block_size=bsize,
                seed=s_val,
                pad_mode=pad_m,
            )
            interleaved_bits, padding_bits, padding_length, block_boundaries, params = interleaver.interleave(
                bits=input_bits,
                seed=seed,
            )
            perm_array = interleaver.permutation
            inv_perm_array = interleaver.inverse_permutation

        else:
            raise ValueError(
                f"Unsupported interleaver type '{chosen_type}'. Available: none, block, convolutional, diagonal, pseudo_random"
            )

        metadata = dict(fec_record.metadata)
        metadata.update({
            "bit_order": self.bit_order,
            "generator_version": self.version,
            "source_fec_record_id": fec_record.fec_record_id,
            "source_frame_id": fec_record.frame_id,
            "source_fec_type": fec_record.fec_type,
            "source_fec_profile": fec_record.fec_profile,
        })

        record = InterleaverRecord(
            interleaver_record_id=iid,
            fec_record_id=fec_record.fec_record_id,
            frame_id=fec_record.frame_id,
            interleaver_type=chosen_type,
            profile_name=chosen_profile,
            input_bits=input_bits,
            interleaved_bits=interleaved_bits,
            input_bit_length=orig_bit_len,
            output_bit_length=len(interleaved_bits),
            padding_bits=padding_bits,
            padding_length=padding_length,
            parameters=params,
            permutation=perm_array,
            inverse_permutation=inv_perm_array,
            block_boundaries=block_boundaries,
            metadata=metadata,
        )

        validate_interleaver_record(record, original_fec=fec_record)
        logger.info(
            "Generated Interleaver record %s for FEC %s (%s, in=%d, out=%d, pad=%d)",
            record.interleaver_record_id,
            record.fec_record_id,
            record.interleaver_type,
            record.input_bit_length,
            record.output_bit_length,
            record.padding_length,
        )
        return record

    def iter_interleaved(
        self,
        fec_records: Iterable[FECRecord],
        interleaver_type: str | None = None,
        profile_name: str | None = None,
        balanced: bool = False,
    ) -> Generator[InterleaverRecord, None, None]:
        """Stream/yield InterleaverRecords one by one.

        Args:
            fec_records: Iterable of input FECRecord objects.
            interleaver_type: Optional fixed interleaver family.
            profile_name: Optional fixed profile.
            balanced: If True, cycles evenly across enabled interleaver families.

        Yields:
            InterleaverRecord objects.
        """
        types_cycle = self.enabled_types.copy() if balanced else None
        idx = 0

        for fec in fec_records:
            if balanced and types_cycle:
                curr_type = types_cycle[idx % len(types_cycle)]
                yield self.interleave(fec_record=fec, interleaver_type=curr_type)
            else:
                yield self.interleave(fec_record=fec, interleaver_type=interleaver_type, profile_name=profile_name)
            idx += 1

    def interleave_batch(
        self,
        fec_records: Sequence[FECRecord],
        interleaver_type: str | None = None,
        profile_name: str | None = None,
        balanced: bool = False,
    ) -> list[InterleaverRecord]:
        """Interleave a sequence of FECRecords into a list of InterleaverRecords."""
        return list(
            self.iter_encoded(fec_records, interleaver_type=interleaver_type, profile_name=profile_name, balanced=balanced)
            if hasattr(self, "iter_encoded")
            else self.iter_interleaved(
                fec_records=fec_records,
                interleaver_type=interleaver_type,
                profile_name=profile_name,
                balanced=balanced,
            )
        )
