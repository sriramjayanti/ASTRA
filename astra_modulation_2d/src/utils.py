"""
Utility functions for ASTRA 2D Spectrogram Classifier.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logger(name: str = "ASTRA_2D", log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures a clean, structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
