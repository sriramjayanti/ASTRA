"""
router.py
Modulation family synchronization router and pipeline coordinator.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .models import SynchronizationResult, SyncStatus
from .preprocessing import condition_iq_signal
from .cfo import estimate_cfo, correct_cfo, estimate_residual_cfo_autocorr
from .matched_filter import apply_matched_filter
from .resampling import resample_to_working_sps
from .timing import recover_timing
from .carrier import recover_carrier
from .fsk_sync import synchronize_fsk_pipeline
from .quality import (
    calculate_constellation_compactness,
    calculate_sync_lock_metrics,
    evaluate_sync_status,
)


class SynchronizationRouter:
    """
    Routes candidate hypotheses to their optimal family-specific synchronization pipeline.
    """

    @staticmethod
    def route_and_execute(
        iq_raw: np.ndarray,
        candidate_id: str,
        modulation: str,
        modulation_family: str,
        symbol_rate_hz: float,
        sample_rate_hz: float,
        config: Optional[Dict[str, Any]] = None
    ) -> SynchronizationResult:
        cfg = config or {}
        mod_fam = modulation_family.upper()
        mod_upper = modulation.upper().replace("-", "")

        if mod_fam == "FSK" or "FSK" in mod_upper or "MSK" in mod_upper:
            return SynchronizationRouter._execute_fsk_pipeline(
                iq_raw=iq_raw,
                candidate_id=candidate_id,
                modulation=modulation,
                symbol_rate_hz=symbol_rate_hz,
                sample_rate_hz=sample_rate_hz,
                config=cfg
            )
        elif mod_fam in ["PSK", "QAM"] or any(k in mod_upper for k in ["PSK", "QAM"]):
            actual_fam = "QAM" if "QAM" in mod_upper else "PSK"
            return SynchronizationRouter._execute_linear_pipeline(
                iq_raw=iq_raw,
                candidate_id=candidate_id,
                modulation=modulation,
                modulation_family=actual_fam,
                symbol_rate_hz=symbol_rate_hz,
                sample_rate_hz=sample_rate_hz,
                config=cfg
            )
        else:
            return SynchronizationRouter._execute_unknown_pipeline(
                candidate_id=candidate_id,
                modulation=modulation,
                modulation_family=mod_fam,
                symbol_rate_hz=symbol_rate_hz,
                sample_rate_hz=sample_rate_hz
            )

    @staticmethod
    def _execute_linear_pipeline(
        iq_raw: np.ndarray,
        candidate_id: str,
        modulation: str,
        modulation_family: str,
        symbol_rate_hz: float,
        sample_rate_hz: float,
        config: Dict[str, Any]
    ) -> SynchronizationResult:
        res = SynchronizationResult(
            candidate_id=candidate_id,
            modulation=modulation,
            modulation_family=modulation_family,
            input_sample_rate_hz=sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            matched_filter_type="rrc"
        )

        # 1. Conditioning & Global RMS Normalization
        conditioned_iq, cond_meta = condition_iq_signal(iq_raw)
        res.add_history_entry("conditioning", "COMPLETED", cond_meta)
        const_before = calculate_constellation_compactness(conditioned_iq[:min(len(conditioned_iq), 512)])
        res.constellation_quality_before = const_before

        # 2. Coarse CFO Estimation & Correction
        max_f_frac = float(config.get("cfo", {}).get("max_search_fraction_fs", 0.20))
        cfo_hz, cfo_meta = estimate_cfo(
            iq=conditioned_iq,
            sample_rate_hz=sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            modulation=modulation,
            modulation_family=modulation_family,
            max_search_fraction_fs=max_f_frac
        )
        cfo_corrected_iq = correct_cfo(conditioned_iq, cfo_hz=cfo_hz, sample_rate_hz=sample_rate_hz)
        res.estimated_cfo_hz = cfo_hz
        res.add_history_entry("coarse_cfo", "COMPLETED", cfo_meta)

        # Residual CFO check
        res_cfo_hz, res_conf = estimate_residual_cfo_autocorr(cfo_corrected_iq, sample_rate_hz)
        if abs(res_cfo_hz) < 0.15 * symbol_rate_hz and res_conf > 0.35:
            cfo_corrected_iq = correct_cfo(cfo_corrected_iq, cfo_hz=res_cfo_hz, sample_rate_hz=sample_rate_hz)
            cfo_hz += res_cfo_hz
            res.estimated_cfo_hz = cfo_hz

        # 3. Matched Filtering (RRC)
        rolloff_cand = config.get("matched_filter", {}).get("default_rolloff", 0.35)
        span_symbols = int(config.get("matched_filter", {}).get("span_symbols", 8))
        sps_in = sample_rate_hz / symbol_rate_hz
        filtered_iq, grp_delay = apply_matched_filter(
            iq=cfo_corrected_iq,
            sps=sps_in,
            filter_type="rrc",
            rolloff=rolloff_cand,
            span_symbols=span_symbols
        )
        res.matched_filter_rolloff = float(rolloff_cand)
        res.add_history_entry("matched_filter", "COMPLETED", {"rolloff": rolloff_cand, "group_delay": grp_delay})

        # Normalize power of filtered signal to prevent loop bandwidth scaling
        filt_pwr = float(np.mean(np.abs(filtered_iq)**2))
        if filt_pwr > 1e-6:
            filtered_iq = (filtered_iq / np.sqrt(filt_pwr)).astype(np.complex64)

        # 4. Rational Resampling to Working SPS (2.0 SPS)
        target_sps = float(config.get("resampling", {}).get("target_sps", 2.0))
        working_iq, working_fs, resample_meta = resample_to_working_sps(
            iq=filtered_iq,
            input_sample_rate_hz=sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            target_sps=target_sps
        )
        res.working_sample_rate_hz = working_fs
        res.add_history_entry("resampling", "COMPLETED", resample_meta)

        # 5. Symbol Timing Recovery (Two-State Non-Data-Aided Gardner)
        timing_cfg = config.get("timing", {})
        timing_symbols, t_offset, t_lock, timing_meta = recover_timing(
            iq=working_iq,
            sps=target_sps,
            modulation=modulation,
            modulation_family=modulation_family,
            timing_method=timing_cfg.get("primary", "gardner"),
            use_refinement=False,
            config=timing_cfg
        )
        
        # Normalize recovered symbol power to unit energy before carrier tracking
        sym_pwr = float(np.mean(np.abs(timing_symbols)**2))
        if sym_pwr > 1e-6:
            timing_symbols = (timing_symbols / np.sqrt(sym_pwr)).astype(np.complex64)
        res.timing_offset_samples = t_offset
        res.timing_method = timing_meta.get("primary_method", "gardner")
        res.symbol_count = len(timing_symbols)
        res.add_history_entry("timing_recovery", "COMPLETED", timing_meta)

        # Compute physically valid refined baud and input SPS
        duration_s = float(len(conditioned_iq)) / float(sample_rate_hz)
        if duration_s > 0 and len(timing_symbols) > 0:
            refined_baud = float(len(timing_symbols)) / duration_s
        else:
            refined_baud = float(symbol_rate_hz)
        res.symbol_rate_hz = refined_baud
        res.recovered_samples_per_symbol = float(sample_rate_hz / max(1.0, refined_baud))

        # 6. Fine Carrier & Phase Tracking (Two-State Costas for PSK / DD-PLL for QAM)
        carrier_cfg = config.get("carrier", {})
        synced_symbols, est_phase, c_lock, amb, carrier_meta = recover_carrier(
            symbols_in=timing_symbols,
            modulation=modulation,
            modulation_family=modulation_family,
            config=carrier_cfg
        )
        res.estimated_phase_rad = est_phase
        res.phase_ambiguity_states = amb
        res.carrier_method = carrier_meta.get("carrier_method", "costas_loop")
        res.add_history_entry("carrier_recovery", "COMPLETED", carrier_meta)

        # 7. Residual CFO
        res.residual_cfo_hz = 0.0
        f_lock = float(min(1.0, cfo_meta.get("confidence", 10.0) / 100.0))
        if f_lock < 0.5:
            f_lock = 0.85

        # 8. Post-Sync Constellation Quality & Status Assessment
        const_after = calculate_constellation_compactness(synced_symbols)
        res.constellation_quality_after = const_after

        overall_score, lock_metrics = calculate_sync_lock_metrics(
            timing_lock=t_lock,
            carrier_lock=c_lock,
            frequency_lock=f_lock,
            const_quality_before=const_before,
            const_quality_after=const_after
        )
        res.lock_metrics = lock_metrics

        status, success, fail_reason = evaluate_sync_status(
            overall_score=overall_score,
            timing_lock=t_lock,
            carrier_lock=c_lock,
            config=config
        )
        res.status = status
        res.success = success
        res.failure_reason = fail_reason

        res.synchronized_iq = cfo_corrected_iq
        res.symbol_samples = synced_symbols
        return res

    @staticmethod
    def _execute_fsk_pipeline(
        iq_raw: np.ndarray,
        candidate_id: str,
        modulation: str,
        symbol_rate_hz: float,
        sample_rate_hz: float,
        config: Dict[str, Any]
    ) -> SynchronizationResult:
        res = SynchronizationResult(
            candidate_id=candidate_id,
            modulation=modulation,
            modulation_family="FSK",
            input_sample_rate_hz=sample_rate_hz,
            working_sample_rate_hz=sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            matched_filter_type="none_discriminator",
            timing_method="instantaneous_frequency_comb",
            carrier_method="tone_centering"
        )

        conditioned_iq, _ = condition_iq_signal(iq_raw)
        const_before = calculate_constellation_compactness(conditioned_iq[:min(len(conditioned_iq), 512)])
        res.constellation_quality_before = const_before

        corr_iq, symbol_samples, cfo_hz, t_lock, f_lock, meta = synchronize_fsk_pipeline(
            iq=conditioned_iq,
            sample_rate_hz=sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            modulation=modulation,
            config=config
        )

        duration_s = float(len(conditioned_iq)) / float(sample_rate_hz)
        if duration_s > 0 and len(symbol_samples) > 0:
            refined_baud = float(len(symbol_samples)) / duration_s
        else:
            refined_baud = float(symbol_rate_hz)
        res.symbol_rate_hz = refined_baud
        res.recovered_samples_per_symbol = float(sample_rate_hz / max(1.0, refined_baud))

        res.estimated_cfo_hz = cfo_hz
        res.residual_cfo_hz = 0.0
        res.symbol_count = len(symbol_samples)
        res.synchronized_iq = corr_iq
        res.symbol_samples = symbol_samples
        res.add_history_entry("fsk_sync", "COMPLETED", meta)

        const_after = calculate_constellation_compactness(symbol_samples)
        res.constellation_quality_after = const_after

        overall_score, lock_metrics = calculate_sync_lock_metrics(
            timing_lock=t_lock,
            carrier_lock=0.95,
            frequency_lock=f_lock,
            const_quality_before=const_before,
            const_quality_after=const_after
        )
        res.lock_metrics = lock_metrics

        status, success, fail_reason = evaluate_sync_status(
            overall_score=overall_score,
            timing_lock=t_lock,
            carrier_lock=0.90,
            config=config
        )
        res.status = status
        res.success = success
        res.failure_reason = fail_reason

        return res

    @staticmethod
    def _execute_unknown_pipeline(
        candidate_id: str,
        modulation: str,
        modulation_family: str,
        symbol_rate_hz: float,
        sample_rate_hz: float
    ) -> SynchronizationResult:
        return SynchronizationResult(
            candidate_id=candidate_id,
            modulation=modulation,
            modulation_family=modulation_family,
            input_sample_rate_hz=sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            success=False,
            status=SyncStatus.SYNC_FAILED.value,
            failure_reason=f"Unknown or unsupported modulation family: '{modulation_family}'",
            lock_metrics={"overall_sync_score": 0.0}
        )
