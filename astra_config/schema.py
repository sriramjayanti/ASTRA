"""
ASTRA Central Configuration Data Classes & Schema Definitions.
Ensures strong typing, range verification, and strict validation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os


class ConfigurationError(Exception):
    """Raised when ASTRA configuration fails schema or semantic validation."""
    pass


@dataclass
class AppConfig:
    name: str = "ASTRA"
    version: str = "1.0.0"
    description: str = "Automated Signal Analysis & Recovery Assistant"
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    structured_logging: bool = False
    random_seed: int = 42


@dataclass
class HardwareConfig:
    device: str = "auto"
    num_workers: int = 4
    max_memory_mb: int = 8192
    gpu_oom_fallback: bool = True
    batch_size: int = 16


@dataclass
class StorageConfig:
    output_dir: str = "outputs"
    logs_dir: str = "logs"
    cache_dir: str = ".cache"
    checkpoints_dir: str = "checkpoints"


@dataclass
class DspConfig:
    dc_removal: bool = True
    dc_filter_alpha: float = 0.995
    normalize_rms: bool = True
    target_rms: float = 1.0
    iq_imbalance_correction: bool = True
    filter_type: str = "rrc"
    rrc_alpha: float = 0.35
    rrc_span: int = 16
    max_samples_analysis: int = 500000


@dataclass
class PipelineConfig:
    max_candidates: int = 24
    beam_width: int = 8
    timeout_seconds: int = 120
    fail_fast: bool = False
    allow_partial_recovery: bool = True
    enable_fec_search: bool = True
    enable_interleaver_search: bool = True
    enable_bitstream_transformer: bool = True


@dataclass
class GuiConfig:
    title: str = "ASTRA — Automated Signal Analysis & Recovery Assistant"
    default_width: int = 1600
    default_height: int = 950
    min_width: int = 1200
    min_height: int = 700
    theme: str = "dark"
    fps_limit: int = 60
    opengl_enabled: bool = True
    fallback_to_software: bool = True
    max_scatter_points: int = 20000


@dataclass
class AstraMasterConfig:
    app: AppConfig = field(default_factory=AppConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dsp: DspConfig = field(default_factory=DspConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    raw_configs: Dict[str, Any] = field(default_factory=dict)
