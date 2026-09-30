"""
ambiguity.py
Phase ambiguity expansion and FSK tone-swap variant generator.
Produces alternate bitstream candidates for downstream FEC/CRC testing.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .models import ConstellationDefinition, DemodulationVariant
from .hard_decision import slice_hard_decisions
from .llr import compute_soft_llrs
from .quality import evaluate_demodulation_quality


def get_configured_rotations(
    modulation: str,
    modulation_family: str,
    config: Optional[Dict[str, Any]] = None
) -> List[float]:
    """Look up supported rotation angles in degrees for a given modulation."""
    cfg = config or {}
    amb_cfg = cfg.get("ambiguity", {})
    if not amb_cfg.get("enabled", True):
        return [0.0]

    mod = modulation.upper()
    if "BPSK" in mod:
        return [float(d) for d in amb_cfg.get("bpsk", {}).get("rotations_deg", [0.0, 180.0])]
    elif "8PSK" in mod:
        return [float(d) for d in amb_cfg.get("psk8", {}).get("rotations_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])]
    elif "QPSK" in mod or "4PSK" in mod or "OQPSK" in mod:
        return [float(d) for d in amb_cfg.get("qpsk", {}).get("rotations_deg", [0.0, 90.0, 180.0, 270.0])]
    elif "QAM" in mod:
        return [float(d) for d in amb_cfg.get("square_qam", {}).get("rotations_deg", [0.0, 90.0, 180.0, 270.0])]
    return [0.0]


def generate_phase_variants(
    symbols: np.ndarray,
    constellation: ConstellationDefinition,
    candidate_id: str,
    modulation: str,
    modulation_family: str,
    noise_var: float,
    llr_mode: str = "max_log",
    config: Optional[Dict[str, Any]] = None
) -> List[DemodulationVariant]:
    """
    Demodulate all valid rotational phase ambiguity orientations.
    """
    rotations = get_configured_rotations(modulation, modulation_family, config)
    variants = []

    for deg in rotations:
        # Rotate received symbols backwards by candidate phase offset
        rot_rad = np.deg2rad(deg)
        rotator = np.exp(-1j * rot_rad)
        rot_symbols = (symbols * rotator).astype(np.complex64)

        # Slice hard decisions and compute LLRs for this rotation
        hard_res = slice_hard_decisions(rot_symbols, constellation)
        soft_res = compute_soft_llrs(rot_symbols, constellation, noise_variance=noise_var, mode=llr_mode)
        qual = evaluate_demodulation_quality(rot_symbols, hard_res, soft_res, noise_var)

        var_id = f"{candidate_id}_rot{int(round(deg))}"
        variant = DemodulationVariant(
            variant_id=var_id,
            ambiguity_type="phase_rotation",
            rotation_deg=deg,
            symbol_count=len(symbols),
            bit_count=len(hard_res.hard_bits),
            hard_bits=hard_res.hard_bits,
            soft_llrs=soft_res.llrs,
            quality=qual
        )
        variants.append(variant)

    return variants
