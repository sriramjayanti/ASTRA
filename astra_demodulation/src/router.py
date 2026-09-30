"""
router.py
Demodulation router directing symbol streams to family-specific demappers.
"""

from typing import Dict, Any, Optional
import numpy as np

from .models import DemodulationResult, DemodStatus
from .psk import demodulate_psk
from .qam import demodulate_qam
from .fsk import demodulate_fsk


class DemodulationRouter:
    """
    Routes symbol streams to the appropriate digital demodulation engine.
    """

    @staticmethod
    def route_and_demodulate(
        symbols: np.ndarray,
        candidate_id: str,
        modulation: str,
        modulation_family: str,
        sample_rate_hz: float = 192000.0,
        symbol_rate_hz: float = 9600.0,
        sync_noise_var: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> DemodulationResult:
        """
        Execute family-specific demodulation pipeline.
        """
        cfg = config or {}
        fam = modulation_family.upper()

        if fam == "PSK":
            return demodulate_psk(
                symbols=symbols,
                candidate_id=candidate_id,
                modulation=modulation,
                sync_noise_var=sync_noise_var,
                config=cfg
            )
        elif fam == "QAM":
            return demodulate_qam(
                symbols=symbols,
                candidate_id=candidate_id,
                modulation=modulation,
                sync_noise_var=sync_noise_var,
                config=cfg
            )
        elif fam == "FSK":
            return demodulate_fsk(
                symbols=symbols,
                candidate_id=candidate_id,
                modulation=modulation,
                sample_rate_hz=sample_rate_hz,
                symbol_rate_hz=symbol_rate_hz,
                sync_noise_var=sync_noise_var,
                config=cfg
            )
        else:
            return DemodulationResult(
                candidate_id=candidate_id,
                modulation=modulation,
                modulation_family=modulation_family,
                success=False,
                status=DemodStatus.DEMOD_FAILED.value,
                failure_reason=f"Unsupported modulation family for digital demodulation: '{modulation_family}'"
            )
