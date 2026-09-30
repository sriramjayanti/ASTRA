"""
ASTRA Central Configuration Loader and Validator.
Safely loads YAML configurations, validates types/ranges, and applies environment overrides.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .schema import (
    AstraMasterConfig,
    AppConfig,
    HardwareConfig,
    StorageConfig,
    DspConfig,
    PipelineConfig,
    GuiConfig,
    ConfigurationError
)


VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_DEVICES = {"auto", "cpu", "cuda", "directml"}


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    """Safely loads a YAML file and guarantees dictionary output."""
    if not path.exists():
        raise ConfigurationError(f"Missing required configuration file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        raise ConfigurationError(f"Failed to parse YAML file {path}: {str(e)}") from e


def load_astra_config(config_dir: Optional[str] = None) -> AstraMasterConfig:
    """
    Loads and validates all ASTRA configurations from the given or default config directory.
    Applies environment variable overrides.
    """
    base_dir = Path(__file__).resolve().parent.parent
    conf_dir = Path(config_dir) if config_dir else Path(os.environ.get("ASTRA_CONFIG_DIR", base_dir / "configs"))

    if not conf_dir.is_dir():
        # Fallback to local configs directory if running from another path
        conf_dir = base_dir / "configs"

    if not conf_dir.is_dir():
        raise ConfigurationError(f"Configuration directory does not exist: {conf_dir}")

    raw_configs: Dict[str, Any] = {}
    for yml_file in conf_dir.glob("*.yaml"):
        raw_configs[yml_file.stem] = _safe_load_yaml(yml_file)

    # 1. Parse App Config
    app_data = raw_configs.get("app", {}).get("app", {})
    hw_data = raw_configs.get("app", {}).get("hardware", {})
    store_data = raw_configs.get("app", {}).get("storage", {})

    log_level = os.environ.get("ASTRA_LOG_LEVEL", app_data.get("log_level", "INFO")).upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigurationError(f"Invalid log_level: {log_level}. Must be one of {VALID_LOG_LEVELS}")

    device = os.environ.get("ASTRA_DEVICE", hw_data.get("device", "auto")).lower()
    if not (device in VALID_DEVICES or device.startswith("cuda:")):
        raise ConfigurationError(f"Invalid device: {device}. Must be one of {VALID_DEVICES} or cuda:N")

    app_cfg = AppConfig(
        name=app_data.get("name", "ASTRA"),
        version=app_data.get("version", "1.0.0"),
        description=app_data.get("description", "Automated Signal Analysis & Recovery Assistant"),
        environment=app_data.get("environment", "production"),
        debug=bool(app_data.get("debug", False)),
        log_level=log_level,
        structured_logging=bool(app_data.get("structured_logging", False)),
        random_seed=int(app_data.get("random_seed", 42))
    )

    hw_cfg = HardwareConfig(
        device=device,
        num_workers=int(hw_data.get("num_workers", 4)),
        max_memory_mb=int(hw_data.get("max_memory_mb", 8192)),
        gpu_oom_fallback=bool(hw_data.get("gpu_oom_fallback", True)),
        batch_size=int(hw_data.get("batch_size", 16))
    )

    store_cfg = StorageConfig(
        output_dir=os.environ.get("ASTRA_OUTPUT_DIR", store_data.get("output_dir", "outputs")),
        logs_dir=os.environ.get("ASTRA_LOG_DIR", store_data.get("logs_dir", "logs")),
        cache_dir=os.environ.get("ASTRA_CACHE_DIR", store_data.get("cache_dir", ".cache")),
        checkpoints_dir=os.environ.get("ASTRA_MODEL_DIR", store_data.get("checkpoints_dir", "checkpoints"))
    )

    # 2. Parse DSP Config
    dsp_data = raw_configs.get("dsp", {}).get("dsp", {})
    dsp_cfg = DspConfig(
        dc_removal=bool(dsp_data.get("dc_removal", True)),
        dc_filter_alpha=float(dsp_data.get("dc_filter_alpha", 0.995)),
        normalize_rms=bool(dsp_data.get("normalize_rms", True)),
        target_rms=float(dsp_data.get("target_rms", 1.0)),
        iq_imbalance_correction=bool(dsp_data.get("iq_imbalance_correction", True)),
        filter_type=str(dsp_data.get("filter_type", "rrc")),
        rrc_alpha=float(dsp_data.get("rrc_alpha", 0.35)),
        rrc_span=int(dsp_data.get("rrc_span", 16)),
        max_samples_analysis=int(dsp_data.get("max_samples_analysis", 500000))
    )

    # 3. Parse Pipeline Config
    pipe_data = raw_configs.get("pipeline", {}).get("pipeline", {})
    max_cands = int(os.environ.get("ASTRA_MAX_CANDIDATES", pipe_data.get("max_candidates", 24)))
    timeout_sec = int(os.environ.get("ASTRA_PIPELINE_TIMEOUT_SEC", pipe_data.get("timeout_seconds", 120)))

    if max_cands <= 0 or max_cands > 256:
        raise ConfigurationError(f"max_candidates must be between 1 and 256, got {max_cands}")
    if timeout_sec <= 0:
        raise ConfigurationError(f"timeout_seconds must be positive, got {timeout_sec}")

    pipe_cfg = PipelineConfig(
        max_candidates=max_cands,
        beam_width=int(pipe_data.get("beam_width", 8)),
        timeout_seconds=timeout_sec,
        fail_fast=bool(pipe_data.get("fail_fast", False)),
        allow_partial_recovery=bool(pipe_data.get("allow_partial_recovery", True)),
        enable_fec_search=bool(pipe_data.get("enable_fec_search", True)),
        enable_interleaver_search=bool(pipe_data.get("enable_interleaver_search", True)),
        enable_bitstream_transformer=bool(pipe_data.get("enable_bitstream_transformer", True))
    )

    # 4. Parse GUI Config
    gui_data = raw_configs.get("gui", {}).get("gui", {})
    gui_cfg = GuiConfig(
        title=str(gui_data.get("title", "ASTRA — Automated Signal Analysis & Recovery Assistant")),
        default_width=int(gui_data.get("window", {}).get("default_width", 1600)),
        default_height=int(gui_data.get("window", {}).get("default_height", 950)),
        min_width=int(gui_data.get("window", {}).get("min_width", 1200)),
        min_height=int(gui_data.get("window", {}).get("min_height", 700)),
        theme=str(gui_data.get("theme", "dark")),
        fps_limit=int(gui_data.get("fps_limit", 60)),
        opengl_enabled=bool(gui_data.get("opengl", {}).get("enabled", True)),
        fallback_to_software=bool(gui_data.get("opengl", {}).get("fallback_to_software", True)),
        max_scatter_points=int(gui_data.get("plots", {}).get("max_scatter_points", 20000))
    )

    return AstraMasterConfig(
        app=app_cfg,
        hardware=hw_cfg,
        storage=store_cfg,
        dsp=dsp_cfg,
        pipeline=pipe_cfg,
        gui=gui_cfg,
        raw_configs=raw_configs
    )


def validate_config_dir(config_dir: Path) -> Dict[str, Any]:
    """Validates existence and schema compliance for all required configuration files."""
    required_configs = [
        "app.yaml", "dsp.yaml", "models.yaml", "pipeline.yaml",
        "synchronization.yaml", "demodulation.yaml", "interleaver.yaml",
        "fec.yaml", "validation.yaml", "scorer.yaml", "bitstream.yaml",
        "transformer.yaml", "explainability.yaml", "gui.yaml"
    ]

    report = {
        "valid": True,
        "files_checked": len(required_configs),
        "missing_files": [],
        "errors": []
    }

    for req in required_configs:
        p = config_dir / req
        if not p.is_file():
            report["valid"] = False
            report["missing_files"].append(req)
        else:
            try:
                _safe_load_yaml(p)
            except Exception as e:
                report["valid"] = False
                report["errors"].append(f"{req}: {str(e)}")

    return report
