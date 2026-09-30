"""
Production-quality Modulation / Clean IQ Waveform Generator for ASTRA (Engine 5).
Maps InterleaverRecord bitstreams to ideal baseband symbols, applies pulse shaping (RRC/FSK),
and synthesizes clean complex64 IQ waveforms with full ground-truth tracking.
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

from ..interleaving.models import InterleaverRecord
from .models import ModulationRecord
from .psk import PSKModulator
from .qam import QAMModulator
from .fsk import FSKModulator
from .validators import validate_modulation_record

logger = logging.getLogger("astra_synthetic.modulation")


BUILTIN_MODULATION_PROFILES: dict[str, dict[str, Any]] = {
    # 1. FSK Families
    "2fsk_cp": {
        "family": "fsk",
        "type": "2fsk",
        "description": "Continuous-Phase 2-FSK (Tone Spacing = 1.0 * Rs)",
        "params": {"continuous_phase": True, "tone_spacing_ratio": 1.0},
    },
    "4fsk_cp": {
        "family": "fsk",
        "type": "4fsk",
        "description": "Continuous-Phase 4-FSK (Tone Spacing = 1.0 * Rs)",
        "params": {"continuous_phase": True, "tone_spacing_ratio": 1.0},
    },
    "msk_cp": {
        "family": "fsk",
        "type": "msk",
        "description": "Minimum Shift Keying (CPFSK h=0.5, Tone Spacing = 0.5 * Rs)",
        "params": {"continuous_phase": True, "tone_spacing_ratio": 0.5},
    },

    # 2. PSK Families
    "bpsk_rrc": {
        "family": "psk",
        "type": "bpsk",
        "description": "Binary PSK with Root Raised Cosine (beta=0.35, span=8)",
        "params": {"pulse_shape": "rrc", "rolloff": 0.35, "span_symbols": 8},
    },
    "qpsk_rrc": {
        "family": "psk",
        "type": "qpsk",
        "description": "Gray-Coded QPSK with Root Raised Cosine (beta=0.35, span=8)",
        "params": {"pulse_shape": "rrc", "rolloff": 0.35, "span_symbols": 8, "mapping": "gray_standard_v1"},
    },
    "8psk_rrc": {
        "family": "psk",
        "type": "8psk",
        "description": "Gray-Coded 8-PSK with Root Raised Cosine (beta=0.35, span=8)",
        "params": {"pulse_shape": "rrc", "rolloff": 0.35, "span_symbols": 8, "mapping": "gray_standard_v1"},
    },
    "dqpsk_rrc": {
        "family": "psk",
        "type": "dqpsk",
        "description": "Differential QPSK with Root Raised Cosine (beta=0.35, span=8)",
        "params": {"pulse_shape": "rrc", "rolloff": 0.35, "span_symbols": 8, "mapping": "gray_standard_v1"},
    },

    # 3. QAM Families
    "16qam_rrc": {
        "family": "qam",
        "type": "16qam",
        "description": "Gray-Coded Square 16-QAM with Root Raised Cosine (beta=0.35, span=8)",
        "params": {"pulse_shape": "rrc", "rolloff": 0.35, "span_symbols": 8, "mapping": "gray_standard_v1"},
    },
    "64qam_rrc": {
        "family": "qam",
        "type": "64qam",
        "description": "Gray-Coded Square 64-QAM with Root Raised Cosine (beta=0.35, span=8)",
        "params": {"pulse_shape": "rrc", "rolloff": 0.35, "span_symbols": 8, "mapping": "gray_standard_v1"},
    },
    "256qam_rrc": {
        "family": "qam",
        "type": "256qam",
        "description": "Gray-Coded Square 256-QAM with Root Raised Cosine (beta=0.35, span=8)",
        "params": {"pulse_shape": "rrc", "rolloff": 0.35, "span_symbols": 8, "mapping": "gray_standard_v1"},
    },
}


DEFAULT_MODULATION_CONFIG: dict[str, Any] = {
    "modulation_generator": {
        "version": "1.0.0",
        "master_seed": 42,
        "sample_rate_hz": 192000.0,
        "default_symbol_rate_hz": 9600.0,
        "symbol_rates": {
            "mode": "random_choice",
            "values": [1200.0, 2400.0, 4800.0, 9600.0, 19200.0, 24000.0],
        },
        "enabled_modulations": [
            "2fsk",
            "4fsk",
            "bpsk",
            "qpsk",
            "8psk",
            "16qam",
            "64qam",
        ],
        "selection_probability": {
            "2fsk": 0.10,
            "4fsk": 0.10,
            "bpsk": 0.15,
            "qpsk": 0.20,
            "8psk": 0.15,
            "16qam": 0.15,
            "64qam": 0.15,
        },
        "default_profiles": {
            "2fsk": "2fsk_cp",
            "4fsk": "4fsk_cp",
            "msk": "msk_cp",
            "bpsk": "bpsk_rrc",
            "qpsk": "qpsk_rrc",
            "8psk": "8psk_rrc",
            "dqpsk": "dqpsk_rrc",
            "16qam": "16qam_rrc",
            "64qam": "64qam_rrc",
            "256qam": "256qam_rrc",
        },
        "psk_qam": {
            "pulse_shape": "rrc",
            "rolloff": {
                "mode": "random_choice",
                "values": [0.20, 0.25, 0.35, 0.50],
            },
            "filter_span_symbols": 8,
        },
        "fsk": {
            "continuous_phase": True,
            "tone_spacing_ratio": {
                "mode": "random_choice",
                "values": [0.5, 1.0, 1.5],
            },
        },
    }
}


class ModulationGenerator:
    """Configurable clean baseband IQ Modulation Generator for ASTRA synthetic pipeline."""

    VERSION = "1.0.0"

    def __init__(
        self,
        config: dict[str, Any] | str | Path | None = None,
        seed: int | None = None,
    ):
        """Initialize ModulationGenerator.

        Args:
            config: Config dict, YAML file path, or None (uses DEFAULT_MODULATION_CONFIG).
            seed: Optional master seed override.
        """
        self.raw_config = self._load_config(config)
        mg_cfg = self.raw_config.get("modulation_generator", self.raw_config)

        self.version = mg_cfg.get("version", self.VERSION)
        self.master_seed = seed if seed is not None else mg_cfg.get("master_seed", 42)
        self.sample_rate = float(mg_cfg.get("sample_rate_hz", 192000.0))
        self.default_symbol_rate = float(mg_cfg.get("default_symbol_rate_hz", 9600.0))

        self.profiles = BUILTIN_MODULATION_PROFILES.copy()
        self._rng = np.random.default_rng(self.master_seed)
        self._record_counter = 0

        self.enabled_modulations = mg_cfg.get(
            "enabled_modulations", ["2fsk", "4fsk", "bpsk", "qpsk", "8psk", "16qam", "64qam"]
        )
        self.selection_prob = mg_cfg.get("selection_probability", {
            "2fsk": 0.10,
            "4fsk": 0.10,
            "bpsk": 0.15,
            "qpsk": 0.20,
            "8psk": 0.15,
            "16qam": 0.15,
            "64qam": 0.15,
        })
        self.default_profiles = mg_cfg.get("default_profiles", {
            "2fsk": "2fsk_cp",
            "4fsk": "4fsk_cp",
            "msk": "msk_cp",
            "bpsk": "bpsk_rrc",
            "qpsk": "qpsk_rrc",
            "8psk": "8psk_rrc",
            "dqpsk": "dqpsk_rrc",
            "16qam": "16qam_rrc",
            "64qam": "64qam_rrc",
            "256qam": "256qam_rrc",
        })

    def _load_config(self, config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        if config is None:
            return DEFAULT_MODULATION_CONFIG.copy()
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

    def _next_modulation_id(self) -> str:
        self._record_counter += 1
        return f"mod_{self._record_counter:06d}"

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.master_seed = seed
        self._rng = np.random.default_rng(self.master_seed)
        self._record_counter = 0

    def modulate(
        self,
        interleaver_record: InterleaverRecord,
        modulation_type: str | None = None,
        profile_name: str | None = None,
        symbol_rate: float | None = None,
        sample_rate: float | None = None,
        seed: int | None = None,
        modulation_record_id: str | None = None,
        **kwargs: Any,
    ) -> ModulationRecord:
        """Modulate an InterleaverRecord into a clean IQ ModulationRecord.

        Args:
            interleaver_record: Source InterleaverRecord (immutable ground truth).
            modulation_type: Canonical type ('2fsk', '4fsk', 'bpsk', 'qpsk', '8psk', '16qam', '64qam').
            profile_name: Built-in profile name (e.g. 'qpsk_rrc', '16qam_rrc', '2fsk_cp').
            symbol_rate: Symbol rate in Baud (defaults to config rate or 9600.0).
            sample_rate: Sample rate in Hz (defaults to 192000.0).
            seed: Seed for random selection if applicable.
            modulation_record_id: Explicit record ID override.
            **kwargs: Override parameters (rolloff, span_symbols, tone_spacing_ratio, etc.).

        Returns:
            Validated ModulationRecord.
        """
        if not isinstance(interleaver_record, InterleaverRecord):
            raise TypeError(f"Expected InterleaverRecord, got {type(interleaver_record).__name__}")

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        mid = modulation_record_id if modulation_record_id is not None else self._next_modulation_id()

        fs = float(sample_rate) if sample_rate is not None else self.sample_rate
        rs = float(symbol_rate) if symbol_rate is not None else self.default_symbol_rate

        # Resolve type and profile
        if profile_name is not None:
            pname = str(profile_name).strip().lower()
            if pname not in BUILTIN_MODULATION_PROFILES:
                avail = ", ".join(BUILTIN_MODULATION_PROFILES.keys())
                raise ValueError(f"Unknown modulation profile '{profile_name}'. Available: {avail}")
            pdef = BUILTIN_MODULATION_PROFILES[pname]
            chosen_type = pdef["type"]
            chosen_profile = pname
            prof_family = pdef["family"]
            prof_params = pdef["params"]
        elif modulation_type is not None:
            chosen_type = str(modulation_type).strip().lower()
            chosen_profile = self.default_profiles.get(chosen_type)
            pdef = BUILTIN_MODULATION_PROFILES.get(chosen_profile, {})
            prof_family = pdef.get("family", "psk" if "psk" in chosen_type else ("qam" if "qam" in chosen_type else "fsk"))
            prof_params = pdef.get("params", {})
        else:
            types = [t for t in self.enabled_modulations if t in self.selection_prob]
            probs = [self.selection_prob[t] for t in types]
            prob_arr = np.array(probs, dtype=np.float64) / sum(probs)
            chosen_type = str(rng.choice(types, p=prob_arr))
            chosen_profile = self.default_profiles.get(chosen_type)
            pdef = BUILTIN_MODULATION_PROFILES.get(chosen_profile, {})
            prof_family = pdef.get("family", "psk")
            prof_params = pdef.get("params", {})

        sps_float = fs / rs
        sps = int(round(sps_float))
        if abs(sps_float - sps) > 1e-4:
            raise ValueError(
                f"Non-integer SPS {sps_float:.4f} is unsupported in Engine 5 v1 (fs={fs}, rs={rs})"
            )

        input_bits = interleaver_record.interleaved_bits.copy()

        # -------------------------------------------------------------
        # Dispatch Family Modulators
        # -------------------------------------------------------------
        if prof_family == "psk":
            phase_offset = kwargs.get("phase_offset", prof_params.get("phase_offset", 0.0))
            mapping_name = kwargs.get("mapping", prof_params.get("mapping", "gray_standard_v1"))
            psh = kwargs.get("pulse_shape", prof_params.get("pulse_shape", "rrc"))
            ro = kwargs.get("rolloff", prof_params.get("rolloff", 0.35))
            span = kwargs.get("span_symbols", prof_params.get("span_symbols", 8))
            pad_m = kwargs.get("pad_mode", "zeros")

            modulator = PSKModulator(
                modulation_type=chosen_type,
                phase_offset=phase_offset,
                mapping_name=mapping_name,
            )
            (
                clean_iq,
                ideal_symbols,
                symbol_indices,
                pad_bits,
                pad_len,
                mapped_len,
                filter_taps,
                group_delay,
                params,
            ) = modulator.modulate(
                bits=input_bits,
                sps=sps,
                pulse_shape=psh,
                rolloff=ro,
                span_symbols=span,
                pad_mode=pad_m,
            )
            m_order = modulator.m
            k_bits = modulator.k
            pulse_shape_name = psh
            rolloff_val = ro if psh == "rrc" else None
            filter_span_val = span if psh == "rrc" else None

        elif prof_family == "qam":
            mapping_name = kwargs.get("mapping", prof_params.get("mapping", "gray_standard_v1"))
            psh = kwargs.get("pulse_shape", prof_params.get("pulse_shape", "rrc"))
            ro = kwargs.get("rolloff", prof_params.get("rolloff", 0.35))
            span = kwargs.get("span_symbols", prof_params.get("span_symbols", 8))
            pad_m = kwargs.get("pad_mode", "zeros")

            modulator = QAMModulator(
                modulation_type=chosen_type,
                mapping_name=mapping_name,
            )
            (
                clean_iq,
                ideal_symbols,
                symbol_indices,
                pad_bits,
                pad_len,
                mapped_len,
                filter_taps,
                group_delay,
                params,
            ) = modulator.modulate(
                bits=input_bits,
                sps=sps,
                pulse_shape=psh,
                rolloff=ro,
                span_symbols=span,
                pad_mode=pad_m,
            )
            m_order = modulator.m
            k_bits = modulator.k
            pulse_shape_name = psh
            rolloff_val = ro if psh == "rrc" else None
            filter_span_val = span if psh == "rrc" else None

        elif prof_family == "fsk":
            cp = kwargs.get("continuous_phase", prof_params.get("continuous_phase", True))
            ts_ratio = kwargs.get("tone_spacing_ratio", prof_params.get("tone_spacing_ratio", 1.0))
            init_phi = kwargs.get("initial_phase", prof_params.get("initial_phase", 0.0))
            pad_m = kwargs.get("pad_mode", "zeros")

            modulator = FSKModulator(
                modulation_type=chosen_type,
                continuous_phase=cp,
                tone_spacing_ratio=ts_ratio,
                initial_phase=init_phi,
            )
            (
                clean_iq,
                ideal_symbols,
                symbol_indices,
                pad_bits,
                pad_len,
                mapped_len,
                params,
            ) = modulator.modulate(
                bits=input_bits,
                symbol_rate=rs,
                sample_rate=fs,
                pad_mode=pad_m,
            )
            m_order = modulator.m
            k_bits = modulator.k
            group_delay = 0
            pulse_shape_name = "fsk_direct"
            rolloff_val = None
            filter_span_val = None

        else:
            raise ValueError(f"Unknown modulation family '{prof_family}'")

        duration = len(clean_iq) / fs
        avg_power = float(np.mean(np.abs(clean_iq) ** 2)) if len(clean_iq) > 0 else 0.0

        metadata = dict(interleaver_record.metadata)
        metadata.update({
            "generator_version": self.version,
            "source_interleaver_record_id": interleaver_record.interleaver_record_id,
            "source_fec_record_id": interleaver_record.fec_record_id,
            "source_frame_id": interleaver_record.frame_id,
            "profile_name": chosen_profile,
        })

        record = ModulationRecord(
            modulation_record_id=mid,
            interleaver_record_id=interleaver_record.interleaver_record_id,
            fec_record_id=interleaver_record.fec_record_id,
            frame_id=interleaver_record.frame_id,
            modulation_type=chosen_type,
            modulation_family=prof_family,
            modulation_order=m_order,
            bits_per_symbol=k_bits,
            input_bits=input_bits,
            mapped_bit_length=mapped_len,
            mapping_padding_bits=pad_bits,
            mapping_padding_length=pad_len,
            symbol_indices=symbol_indices,
            ideal_symbols=ideal_symbols,
            symbol_count=len(symbol_indices),
            symbol_rate=rs,
            sample_rate=fs,
            samples_per_symbol=float(sps),
            pulse_shape=pulse_shape_name,
            rolloff=rolloff_val,
            filter_span_symbols=filter_span_val,
            filter_group_delay_samples=group_delay,
            clean_iq=clean_iq,
            clean_iq_sample_count=len(clean_iq),
            duration_seconds=duration,
            average_iq_power=avg_power,
            parameters=params,
            metadata=metadata,
        )

        validate_modulation_record(record, original_interleaver=interleaver_record)
        logger.info(
            "Generated Modulation record %s for Interleaver %s (%s, M=%d, syms=%d, IQ_samples=%d)",
            record.modulation_record_id,
            record.interleaver_record_id,
            record.modulation_type,
            record.modulation_order,
            record.symbol_count,
            record.clean_iq_sample_count,
        )
        return record

    def iter_modulated(
        self,
        interleaver_records: Iterable[InterleaverRecord],
        modulation_type: str | None = None,
        profile_name: str | None = None,
        symbol_rate: float | None = None,
        sample_rate: float | None = None,
        balanced: bool = False,
    ) -> Generator[ModulationRecord, None, None]:
        """Stream/yield ModulationRecords one by one."""
        types_cycle = self.enabled_modulations.copy() if balanced else None
        idx = 0

        for int_rec in interleaver_records:
            if balanced and types_cycle:
                curr_type = types_cycle[idx % len(types_cycle)]
                yield self.modulate(
                    interleaver_record=int_rec,
                    modulation_type=curr_type,
                    symbol_rate=symbol_rate,
                    sample_rate=sample_rate,
                )
            else:
                yield self.modulate(
                    interleaver_record=int_rec,
                    modulation_type=modulation_type,
                    profile_name=profile_name,
                    symbol_rate=symbol_rate,
                    sample_rate=sample_rate,
                )
            idx += 1

    def modulate_batch(
        self,
        interleaver_records: Sequence[InterleaverRecord],
        modulation_type: str | None = None,
        profile_name: str | None = None,
        symbol_rate: float | None = None,
        sample_rate: float | None = None,
        balanced: bool = False,
    ) -> list[ModulationRecord]:
        """Modulate a sequence of InterleaverRecords into a list of ModulationRecords."""
        return list(
            self.iter_modulated(
                interleaver_records=interleaver_records,
                modulation_type=modulation_type,
                profile_name=profile_name,
                symbol_rate=symbol_rate,
                sample_rate=sample_rate,
                balanced=balanced,
            )
        )
