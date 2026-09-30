"""
ASTRA Synthetic Engine 6: RF / Channel Impairment Generator.
Simulates realistic propagation phenomena: AWGN, CFO, phase offsets,
fractional timing shifts, gain scaling, frequency drift, Rayleigh/Rician fading,
multipath tapped-delay line channels, sinusoidal interference, and sample clock offsets.
"""

from .models import ChannelRecord
from .generator import ChannelGenerator
from .awgn import apply_awgn
from .cfo import apply_cfo
from .phase import apply_phase_offset
from .timing import apply_timing_offset
from .gain import apply_gain
from .drift import apply_frequency_drift
from .fading import apply_rayleigh_fading, apply_rician_fading
from .multipath import apply_multipath, MULTIPATH_PROFILES
from .interference import apply_sinusoidal_interference
from .clock_offset import apply_sample_clock_offset
from .validators import validate_channel_record
from .serializers import save_channel_record, load_channel_record, save_channel_batch

__all__ = [
    "ChannelRecord",
    "ChannelGenerator",
    "apply_awgn",
    "apply_cfo",
    "apply_phase_offset",
    "apply_timing_offset",
    "apply_gain",
    "apply_frequency_drift",
    "apply_rayleigh_fading",
    "apply_rician_fading",
    "apply_multipath",
    "MULTIPATH_PROFILES",
    "apply_sinusoidal_interference",
    "apply_sample_clock_offset",
    "validate_channel_record",
    "save_channel_record",
    "load_channel_record",
    "save_channel_batch",
]
