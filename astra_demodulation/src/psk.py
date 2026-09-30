"""
psk.py
PSK Family Demodulator for BPSK, QPSK, and 8PSK with global power normalization.
"""

from typing import Dict, Any, Optional
import numpy as np

from .models import DemodulationResult
from .mappings import get_constellation
from .hard_decision import slice_hard_decisions
from .llr import compute_soft_llrs
from .noise import estimate_noise_variance
from .quality import evaluate_demodulation_quality, determine_demod_status
from .ambiguity import generate_phase_variants


def demodulate_psk(
    symbols: np.ndarray,
    candidate_id: str,
    modulation: str = "QPSK",
    sync_noise_var: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> DemodulationResult:
    """
    Demodulate PSK symbols (BPSK, QPSK, 8PSK) into hard bits, soft LLRs, and phase ambiguity variants.
    """
    cfg = config or {}
    constellation = get_constellation(modulation)
    llr_mode = cfg.get("llr", {}).get("mode", "max_log")

    # Global constellation energy normalization
    current_pwr = float(np.mean(np.abs(symbols) ** 2))
    if current_pwr > 1e-6 and abs(current_pwr - 1.0) > 0.05:
        scale = 1.0 / np.sqrt(current_pwr)
        scaled_symbols = (symbols * scale).astype(np.complex64)
    else:
        scaled_symbols = symbols.copy()

    # 1. Nominal Hard Decisions
    hard_res = slice_hard_decisions(scaled_symbols, constellation)

    # 2. Noise Variance
    noise_var, noise_source = estimate_noise_variance(scaled_symbols, hard_res.nearest_points, sync_noise_var)

    # 3. Soft LLR Computation
    soft_res = compute_soft_llrs(scaled_symbols, constellation, noise_variance=noise_var, mode=llr_mode)

    # 4. Telemetry and EVM Evaluation
    quality = evaluate_demodulation_quality(scaled_symbols, hard_res, soft_res, noise_var)
    status, success, fail_reason = determine_demod_status(quality, len(symbols), cfg)

    # 5. Rotational Phase Ambiguity Variants
    variants = generate_phase_variants(
        symbols=scaled_symbols,
        constellation=constellation,
        candidate_id=candidate_id,
        modulation=modulation,
        modulation_family="PSK",
        noise_var=noise_var,
        llr_mode=llr_mode,
        config=cfg
    )

    result = DemodulationResult(
        candidate_id=candidate_id,
        modulation=modulation,
        modulation_family="PSK",
        success=success,
        status=status,
        symbol_count=len(symbols),
        bits_per_symbol=constellation.bits_per_symbol,
        bit_count=len(hard_res.hard_bits),
        hard_symbol_indices=hard_res.symbol_indices,
        hard_bits=hard_res.hard_bits,
        soft_llrs=soft_res.llrs,
        phase_variants=variants,
        quality=quality,
        noise_variance=noise_var,
        noise_source=noise_source,
        failure_reason=fail_reason
    )
    result.add_history_entry("psk_demodulation", status, quality.to_dict())
    return result
