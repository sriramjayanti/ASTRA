"""
timing.py
Unified timing recovery router and coordinator.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from .gardner import gardner_recover
from .mueller_muller import mueller_muller_recover
from .models import TimingState


def recover_timing(
    iq: np.ndarray,
    sps: float = 2.0,
    modulation: str = "QPSK",
    modulation_family: str = "PSK",
    timing_method: str = "gardner",
    use_refinement: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Perform symbol timing recovery and extract 1-sample-per-symbol on-time constellation points.
    
    Returns:
        (symbol_samples, timing_offset, timing_lock_score, metadata)
    """
    cfg = config or {}
    kp = float(cfg.get("gardner_loop_gain", 0.015))
    
    # 1. Primary Timing Recovery (Non-Data-Aided Gardner)
    symbols, timing_offset, lock_score, state = gardner_recover(
        iq=iq,
        sps=sps,
        loop_gain_kp=kp
    )

    refined = False
    # 2. Optional Decision-Directed M&M Refinement for PSK/QAM when Gardner has reasonable lock
    if use_refinement and lock_score > 0.4 and modulation_family in ["PSK", "QAM"] and len(symbols) > 16:
        mm_kp = float(cfg.get("mm_loop_gain", 0.01))
        symbols_refined, _, mm_lock, _ = mueller_muller_recover(
            symbols_in=symbols,
            modulation=modulation,
            loop_gain_kp=mm_kp
        )
        if mm_lock >= lock_score:
            symbols = symbols_refined
            lock_score = mm_lock
            refined = True

    meta = {
        "primary_method": timing_method,
        "refined_with_mm": refined,
        "timing_lock_score": float(lock_score),
        "timing_offset_samples": float(timing_offset),
        "symbol_count": len(symbols)
    }

    return symbols, timing_offset, lock_score, meta
