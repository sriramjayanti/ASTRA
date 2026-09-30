"""
fsk.py
FSK Family Demodulator for 2-FSK and 4-FSK waveforms.
"""

from typing import Dict, Any, Optional, List
import numpy as np

from .models import DemodulationResult, DemodulationVariant, DemodulationQuality, DemodStatus
from .noise import estimate_noise_variance


def demodulate_fsk(
    symbols: np.ndarray,
    candidate_id: str,
    modulation: str = "2-FSK",
    sample_rate_hz: float = 192000.0,
    symbol_rate_hz: float = 9600.0,
    sync_noise_var: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> DemodulationResult:
    """
    Demodulate 2-FSK and 4-FSK waveforms into hard bits, soft LLRs, and tone-swap variants.
    """
    cfg = config or {}
    is_4fsk = "4" in modulation
    bps = 2 if is_4fsk else 1
    n_symbols = len(symbols)

    if n_symbols < 2:
        return DemodulationResult(
            candidate_id=candidate_id,
            modulation=modulation,
            modulation_family="FSK",
            success=False,
            status=DemodStatus.DEMOD_FAILED.value,
            failure_reason="Insufficient FSK symbols"
        )

    # 1. Extract instantaneous phase angle / frequency deviations
    phase_diffs = np.angle(symbols[1:] * np.conj(symbols[:-1]))
    phase_diffs = np.concatenate([[phase_diffs[0]], phase_diffs])

    # Normalize tone deviations
    std_dev = np.std(phase_diffs) + 1e-6
    norm_dev = (phase_diffs - np.mean(phase_diffs)) / std_dev

    noise_var = sync_noise_var if (sync_noise_var and sync_noise_var > 1e-6) else 0.05

    if not is_4fsk:
        # 2-FSK: positive -> bit 0, negative -> bit 1
        hard_symbols = np.where(norm_dev >= 0, 0, 1).astype(np.int32)
        hard_bits = hard_symbols.astype(np.uint8)
        
        # LLR = 2 * norm_dev / noise_var (positive -> 0, negative -> 1)
        llrs = np.clip((2.0 * norm_dev / noise_var), -30.0, 30.0).astype(np.float32)
    else:
        # 4-FSK: 4 levels [-1.5, -0.5, +0.5, +1.5]
        # Gray mapping:
        # <= -1.0       -> Tone 0 -> 00
        # -1.0 to 0.0   -> Tone 1 -> 01
        # 0.0 to 1.0    -> Tone 2 -> 11
        # >= 1.0        -> Tone 3 -> 10
        tone_indices = np.zeros(n_symbols, dtype=np.int32)
        tone_indices[norm_dev < -1.0] = 0
        tone_indices[(norm_dev >= -1.0) & (norm_dev < 0.0)] = 1
        tone_indices[(norm_dev >= 0.0) & (norm_dev < 1.0)] = 2
        tone_indices[norm_dev >= 1.0] = 3

        gray_4fsk_map = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
        hard_bits = gray_4fsk_map[tone_indices].flatten()
        hard_symbols = tone_indices

        # Soft LLRs for bit 0 and bit 1
        # Bit 0 is 0 for Tones 0, 1 (< 0) and 1 for Tones 2, 3 (>= 0)
        llr_b0 = -norm_dev * 2.0 / noise_var
        # Bit 1 is 0 for Tones 0, 3 (|norm_dev| >= 1.0) and 1 for Tones 1, 2 (|norm_dev| < 1.0)
        llr_b1 = (np.abs(norm_dev) - 1.0) * 2.0 / noise_var
        
        llrs = np.empty(n_symbols * 2, dtype=np.float32)
        llrs[0::2] = np.clip(llr_b0, -30.0, 30.0)
        llrs[1::2] = np.clip(llr_b1, -30.0, 30.0)

    # Telemetry
    evm_rms = float(np.std(norm_dev - np.round(norm_dev)))
    quality = DemodulationQuality(
        evm_rms=evm_rms,
        evm_percent=float(evm_rms * 100.0),
        evm_db=float(20.0 * np.log10(max(1e-4, evm_rms))),
        snr_estimate_db=float(-10.0 * np.log10(noise_var)),
        mean_decision_distance=float(np.mean(np.abs(norm_dev))),
        mean_decision_margin=0.5,
        mean_abs_llr=float(np.mean(np.abs(llrs))),
        low_confidence_bit_fraction=float(np.mean(np.abs(llrs) < 1.0)),
        demodulation_quality_score=float(max(0.0, min(1.0, 1.0 / (1.0 + evm_rms))))
    )

    # Variants: Normal and Swapped Tone Polarity
    variants = []
    nominal_var = DemodulationVariant(
        variant_id=f"{candidate_id}_tone_norm",
        ambiguity_type="tone_polarity",
        rotation_deg=0.0,
        symbol_count=n_symbols,
        bit_count=len(hard_bits),
        hard_bits=hard_bits,
        soft_llrs=llrs,
        quality=quality
    )
    variants.append(nominal_var)

    if cfg.get("ambiguity", {}).get("fsk", {}).get("allow_tone_label_swap", True):
        # Inverted tone polarity variant
        swapped_bits = (1 - hard_bits).astype(np.uint8)
        swapped_llrs = (-llrs).astype(np.float32)
        swap_var = DemodulationVariant(
            variant_id=f"{candidate_id}_tone_swap",
            ambiguity_type="tone_polarity",
            rotation_deg=180.0,
            symbol_count=n_symbols,
            bit_count=len(swapped_bits),
            hard_bits=swapped_bits,
            soft_llrs=swapped_llrs,
            quality=quality
        )
        variants.append(swap_var)

    result = DemodulationResult(
        candidate_id=candidate_id,
        modulation=modulation,
        modulation_family="FSK",
        success=True,
        status=DemodStatus.DEMOD_PASSED.value,
        symbol_count=n_symbols,
        bits_per_symbol=bps,
        bit_count=len(hard_bits),
        hard_symbol_indices=hard_symbols,
        hard_bits=hard_bits,
        soft_llrs=llrs,
        phase_variants=variants,
        quality=quality,
        noise_variance=noise_var,
        noise_source="instantaneous_frequency"
    )
    result.add_history_entry("fsk_demodulation", "DEMOD_PASSED", quality.to_dict())
    return result
