"""
ASTRA Stage 7 — Demodulation Engine
"""

from .models import (
    DemodulationResult,
    DemodulationVariant,
    ConstellationDefinition,
    HardDecisionResult,
    SoftDecisionResult,
    DemodulationQuality,
    DemodStatus,
)
from .inference import DemodulationEngine
from .router import DemodulationRouter
from .mappings import get_constellation, CANONICAL_CONSTELLATIONS
from .hard_decision import slice_hard_decisions
from .llr import compute_soft_llrs
from .psk import demodulate_psk
from .qam import demodulate_qam
from .fsk import demodulate_fsk
from .ambiguity import generate_phase_variants
from .evm import compute_evm
from .noise import estimate_noise_variance
from .quality import evaluate_demodulation_quality, determine_demod_status
from .utils import generate_synthetic_demod_test_data

__all__ = [
    "DemodulationEngine",
    "DemodulationResult",
    "DemodulationVariant",
    "ConstellationDefinition",
    "HardDecisionResult",
    "SoftDecisionResult",
    "DemodulationQuality",
    "DemodStatus",
    "DemodulationRouter",
    "get_constellation",
    "CANONICAL_CONSTELLATIONS",
    "slice_hard_decisions",
    "compute_soft_llrs",
    "demodulate_psk",
    "demodulate_qam",
    "demodulate_fsk",
    "generate_phase_variants",
    "compute_evm",
    "estimate_noise_variance",
    "evaluate_demodulation_quality",
    "determine_demod_status",
    "generate_synthetic_demod_test_data",
]
