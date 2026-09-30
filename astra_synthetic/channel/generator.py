"""
Main Channel / RF Impairment Generator for ASTRA (Engine 6).
Applies configurable physical channel effects to pristine ModulationRecord IQ,
recording exact ground truth and telemetry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator, Iterable
import numpy as np
import yaml

from astra_synthetic.modulation.models import ModulationRecord
from .models import ChannelRecord
from .awgn import apply_awgn
from .cfo import apply_cfo
from .phase import apply_phase_offset
from .timing import apply_timing_offset
from .gain import apply_gain
from .drift import apply_frequency_drift
from .fading import apply_rayleigh_fading, apply_rician_fading
from .multipath import apply_multipath
from .interference import apply_sinusoidal_interference
from .clock_offset import apply_sample_clock_offset
from .validators import validate_channel_record

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "channel_config.yaml"


class ChannelGenerator:
    """Orchestrates realistic baseband RF impairments while preserving exact ground truth."""

    CANONICAL_ORDER = [
        "gain",
        "fading",
        "multipath",
        "clock_offset",
        "timing_offset",
        "cfo",
        "phase_offset",
        "frequency_drift",
        "interference",
        "awgn",
    ]

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

        gen_cfg = self.config.get("channel_generator", {})
        self.version = gen_cfg.get("version", "1.0.0")
        self.master_seed = gen_cfg.get("master_seed", 42)
        self.profiles = self.config.get("profiles", {})
        self._record_counter = 0

    def apply(
        self,
        modulation_record: ModulationRecord,
        profile_name: str | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> ChannelRecord:
        """Apply RF channel impairments to a single ModulationRecord."""
        self._record_counter += 1
        chan_id = f"chan_{self._record_counter:06d}"

        # Setup deterministic RNG
        if seed is None:
            seed = (self.master_seed + self._record_counter * 10007) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        # Merge profile & overrides
        profile = {}
        if profile_name and profile_name in self.profiles:
            profile = dict(self.profiles[profile_name])
        elif profile_name:
            logger.warning("Profile '%s' not found in configuration; using defaults.", profile_name)

        params = dict(profile)
        if overrides:
            params.update(overrides)

        clean_iq = np.asarray(modulation_record.clean_iq, dtype=np.complex64)
        Fs = float(modulation_record.sample_rate)
        sps = float(modulation_record.samples_per_symbol)

        current_iq = clean_iq.copy()
        power_stages: dict[str, float] = {}
        power_stages["clean_signal"] = float(np.mean(np.abs(clean_iq) ** 2)) if len(clean_iq) > 0 else 0.0

        applied_order = []

        # 1. Gain Scaling
        gain_db = 0.0
        gain_linear = 1.0
        if params.get("gain_db") is not None:
            g_cfg = params["gain_db"]
            if isinstance(g_cfg, (int, float)):
                gain_db = float(g_cfg)
            elif isinstance(g_cfg, dict) and "range" in g_cfg:
                gain_db = float(rng.uniform(g_cfg["range"][0], g_cfg["range"][1]))
            elif isinstance(g_cfg, dict) and "values" in g_cfg:
                gain_db = float(rng.choice(g_cfg["values"]))

            current_iq, gain_db, gain_linear = apply_gain(current_iq, gain_db=gain_db)
            power_stages["after_gain"] = float(np.mean(np.abs(current_iq) ** 2))
            applied_order.append("gain")

        # 2. Fading (Rayleigh / Rician)
        fading_type = "none"
        fading_params: dict[str, Any] = {}
        f_cfg = params.get("fading")
        if f_cfg and isinstance(f_cfg, dict) and f_cfg.get("type", "none") != "none":
            fading_type = f_cfg.get("type", "none").lower()
            if fading_type == "rayleigh":
                current_iq, h_coeff = apply_rayleigh_fading(current_iq, seed=rng.integers(0, 2**31 - 1))
                fading_params["fading_coefficient"] = (float(h_coeff.real), float(h_coeff.imag))
                power_stages["after_fading"] = float(np.mean(np.abs(current_iq) ** 2))
                applied_order.append("fading")
            elif fading_type == "rician":
                k_cfg = f_cfg.get("k_factor_db", 10.0)
                if isinstance(k_cfg, (int, float)):
                    k_db = float(k_cfg)
                elif isinstance(k_cfg, list):
                    k_db = float(rng.choice(k_cfg))
                elif isinstance(k_cfg, dict) and "range" in k_cfg:
                    k_db = float(rng.uniform(k_cfg["range"][0], k_cfg["range"][1]))
                else:
                    k_db = 10.0
                current_iq, h_coeff, k_db_used = apply_rician_fading(
                    current_iq, k_factor_db=k_db, seed=rng.integers(0, 2**31 - 1)
                )
                fading_params["k_factor_db"] = k_db_used
                fading_params["fading_coefficient"] = (float(h_coeff.real), float(h_coeff.imag))
                power_stages["after_fading"] = float(np.mean(np.abs(current_iq) ** 2))
                applied_order.append("fading")

        # 3. Multipath Channel
        multipath_enabled = False
        mp_taps_used = None
        mp_delays_used = None
        mp_cfg = params.get("multipath")
        if mp_cfg and isinstance(mp_cfg, dict) and mp_cfg.get("enabled", True):
            multipath_enabled = True
            mp_prof = mp_cfg.get("profile", "mild")
            custom_delays = mp_cfg.get("delays_samples")
            custom_taps = mp_cfg.get("taps")
            if custom_taps is not None:
                custom_taps = np.array(
                    [complex(r, i) if isinstance(r, (int, float)) else t for t, (r, i) in zip(custom_taps, custom_taps)],
                    dtype=np.complex64,
                )
            current_iq, mp_taps_used, mp_delays_used = apply_multipath(
                current_iq,
                profile=mp_prof,
                delays_samples=custom_delays,
                taps=custom_taps,
                normalize=mp_cfg.get("normalize", True),
            )
            power_stages["after_multipath"] = float(np.mean(np.abs(current_iq) ** 2))
            applied_order.append("multipath")

        # 4. Sample Clock Offset (ppm)
        sample_clock_offset_ppm = 0.0
        if params.get("sample_clock_offset_ppm") is not None:
            clk_cfg = params["sample_clock_offset_ppm"]
            if isinstance(clk_cfg, (int, float)):
                sample_clock_offset_ppm = float(clk_cfg)
            elif isinstance(clk_cfg, dict) and "range" in clk_cfg:
                sample_clock_offset_ppm = float(rng.uniform(clk_cfg["range"][0], clk_cfg["range"][1]))

            if abs(sample_clock_offset_ppm) > 1e-6:
                current_iq, _ = apply_sample_clock_offset(
                    current_iq, ppm=sample_clock_offset_ppm, sample_rate=Fs
                )
                power_stages["after_clock_offset"] = float(np.mean(np.abs(current_iq) ** 2))
                applied_order.append("clock_offset")

        # 5. Timing Offset
        timing_offset_samples = 0.0
        if params.get("timing_offset_samples") is not None:
            t_cfg = params["timing_offset_samples"]
            if isinstance(t_cfg, (int, float)):
                timing_offset_samples = float(t_cfg)
            elif isinstance(t_cfg, dict) and "range" in t_cfg:
                timing_offset_samples = float(rng.uniform(t_cfg["range"][0], t_cfg["range"][1]))
            elif isinstance(t_cfg, dict) and "values" in t_cfg:
                timing_offset_samples = float(rng.choice(t_cfg["values"]))

            if abs(timing_offset_samples) > 1e-6:
                current_iq = apply_timing_offset(current_iq, timing_offset_samples=timing_offset_samples)
                power_stages["after_timing"] = float(np.mean(np.abs(current_iq) ** 2))
                applied_order.append("timing_offset")

        timing_offset_symbols = timing_offset_samples / sps if sps > 0 else 0.0

        # 6. Carrier Frequency Offset (CFO)
        cfo_hz = 0.0
        if params.get("cfo_hz") is not None:
            cfo_cfg = params["cfo_hz"]
            if isinstance(cfo_cfg, (int, float)):
                cfo_hz = float(cfo_cfg)
            elif isinstance(cfo_cfg, dict) and "range" in cfo_cfg:
                cfo_hz = float(rng.uniform(cfo_cfg["range"][0], cfo_cfg["range"][1]))
            elif isinstance(cfo_cfg, dict) and "values" in cfo_cfg:
                cfo_hz = float(rng.choice(cfo_cfg["values"]))

            if abs(cfo_hz) > 1e-6:
                current_iq = apply_cfo(current_iq, cfo_hz=cfo_hz, sample_rate=Fs)
                power_stages["after_cfo"] = float(np.mean(np.abs(current_iq) ** 2))
                applied_order.append("cfo")

        # 7. Carrier Phase Offset
        phase_offset_rad = 0.0
        if params.get("phase_offset_rad") is not None:
            p_cfg = params["phase_offset_rad"]
            if isinstance(p_cfg, (int, float)):
                phase_offset_rad = float(p_cfg)
            elif isinstance(p_cfg, dict) and "range" in p_cfg:
                phase_offset_rad = float(rng.uniform(p_cfg["range"][0], p_cfg["range"][1]))
            elif isinstance(p_cfg, dict) and "values" in p_cfg:
                phase_offset_rad = float(rng.choice(p_cfg["values"]))

            if abs(phase_offset_rad) > 1e-6:
                current_iq = apply_phase_offset(current_iq, phase_offset_rad=phase_offset_rad)
                power_stages["after_phase"] = float(np.mean(np.abs(current_iq) ** 2))
                applied_order.append("phase_offset")

        # 8. Frequency Drift
        frequency_drift_hz_per_sec = 0.0
        if params.get("frequency_drift_hz_per_sec") is not None:
            d_cfg = params["frequency_drift_hz_per_sec"]
            if isinstance(d_cfg, (int, float)):
                frequency_drift_hz_per_sec = float(d_cfg)
            elif isinstance(d_cfg, dict) and "range" in d_cfg:
                frequency_drift_hz_per_sec = float(rng.uniform(d_cfg["range"][0], d_cfg["range"][1]))

            if abs(frequency_drift_hz_per_sec) > 1e-6:
                current_iq, _ = apply_frequency_drift(
                    current_iq,
                    drift_hz_per_sec=frequency_drift_hz_per_sec,
                    sample_rate=Fs,
                    initial_cfo_hz=0.0,
                )
                power_stages["after_drift"] = float(np.mean(np.abs(current_iq) ** 2))
                applied_order.append("frequency_drift")

        # 9. Interference (Narrowband/Tone)
        interference_enabled = False
        interference_params: dict[str, Any] = {}
        int_cfg = params.get("interference")
        if int_cfg and isinstance(int_cfg, dict) and int_cfg.get("enabled", False):
            interference_enabled = True
            current_iq, _, int_info = apply_sinusoidal_interference(
                current_iq,
                sample_rate=Fs,
                frequency_offset_hz=int_cfg.get("freq_offset_hz", 1000.0),
                power_relative_db=int_cfg.get("power_relative_db", -10.0),
                initial_phase_rad=int_cfg.get("phase_rad", 0.0),
            )
            interference_params = int_info
            power_stages["after_interference"] = float(np.mean(np.abs(current_iq) ** 2))
            applied_order.append("interference")

        # 10. AWGN (Applied last to maintain exact SNR relative to received signal power)
        snr_db_target = None
        snr_db_measured = None
        if params.get("snr_db") is not None:
            snr_cfg = params["snr_db"]
            if isinstance(snr_cfg, (int, float)):
                snr_db_target = float(snr_cfg)
            elif isinstance(snr_cfg, dict) and "range" in snr_cfg:
                snr_db_target = float(rng.uniform(snr_cfg["range"][0], snr_cfg["range"][1]))
            elif isinstance(snr_cfg, dict) and "values" in snr_cfg:
                snr_db_target = float(rng.choice(snr_cfg["values"]))

            if snr_db_target is not None:
                current_iq, p_sig, p_noise, snr_meas = apply_awgn(
                    current_iq,
                    snr_db=snr_db_target,
                    seed=rng.integers(0, 2**31 - 1),
                )
                snr_db_measured = snr_meas
                power_stages["noise_power"] = float(p_noise)
                power_stages["final_power"] = float(np.mean(np.abs(current_iq) ** 2))
                applied_order.append("awgn")
        else:
            power_stages["final_power"] = float(np.mean(np.abs(current_iq) ** 2)) if len(current_iq) > 0 else 0.0

        metadata = {
            "generator": "ASTRA Channel Impairment Generator",
            "version": self.version,
            "seed": seed,
            "profile_name": profile_name,
            "modulation_type": getattr(modulation_record, "modulation_type", "unknown"),
        }

        record = ChannelRecord(
            channel_record_id=chan_id,
            modulation_record_id=modulation_record.modulation_record_id,
            interleaver_record_id=modulation_record.interleaver_record_id,
            fec_record_id=modulation_record.fec_record_id,
            frame_id=modulation_record.frame_id,
            clean_iq=clean_iq,
            impaired_iq=current_iq,
            sample_rate=Fs,
            snr_db_target=snr_db_target,
            snr_db_measured=snr_db_measured,
            cfo_hz=cfo_hz,
            phase_offset_rad=phase_offset_rad,
            timing_offset_samples=timing_offset_samples,
            timing_offset_symbols=timing_offset_symbols,
            gain_db=gain_db,
            gain_linear=gain_linear,
            frequency_drift_hz_per_sec=frequency_drift_hz_per_sec,
            fading_type=fading_type,
            fading_parameters=fading_params,
            multipath_enabled=multipath_enabled,
            multipath_taps=mp_taps_used,
            multipath_delays_samples=mp_delays_used,
            interference_enabled=interference_enabled,
            interference_parameters=interference_params,
            sample_clock_offset_ppm=sample_clock_offset_ppm,
            power_stages=power_stages,
            impairment_order=applied_order,
            parameters=params,
            metadata=metadata,
        )

        validate_channel_record(record)
        return record

    def apply_batch(
        self,
        modulation_records: list[ModulationRecord],
        profile_name: str | None = None,
        base_seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> list[ChannelRecord]:
        """Apply channel impairments to a list of ModulationRecords."""
        return list(
            self.iter_impaired(
                modulation_records,
                profile_name=profile_name,
                base_seed=base_seed,
                overrides=overrides,
            )
        )

    def iter_impaired(
        self,
        modulation_records: Iterable[ModulationRecord],
        profile_name: str | None = None,
        base_seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Generator[ChannelRecord, None, None]:
        """Stream impaired ChannelRecords for large dataset generation."""
        seed = base_seed if base_seed is not None else self.master_seed
        for idx, mod_rec in enumerate(modulation_records):
            item_seed = (seed + idx * 7919) & 0xFFFFFFFF
            yield self.apply(
                mod_rec,
                profile_name=profile_name,
                seed=item_seed,
                overrides=overrides,
            )
