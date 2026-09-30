"""
ASTRA Stage 15: Provenance and Model Version Tracking.

Ensures strict reproducibility, auditing, and configuration tracking.
"""

import hashlib
import json
from typing import Dict, Any


class ProvenanceTracker:
    """Tracks model versions, config hashes, and execution provenance."""

    DEFAULT_VERSIONS = {
        "resnet_1d": "1.2.0-quantized",
        "cnn_2d": "1.1.4-stft",
        "fusion_engine": "2.0.1-weighted",
        "random_forest": "1.0.0-rf-dsp",
        "symbol_rate_xgboost": "1.3.2",
        "pipeline_xgboost_scorer": "1.5.0-calibrated",
        "bitstream_transformer": "1.0.0-1dcnn-trans",
        "payload_explorer": "1.0.0",
        "explainability_engine": "1.0.0"
    }

    def get_provenance_record(
        self,
        config: Dict[str, Any],
        custom_versions: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Returns complete provenance metadata including config hash."""
        versions = dict(self.DEFAULT_VERSIONS)
        if custom_versions:
            versions.update(custom_versions)

        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

        return {
            "engine": "ASTRA Stage 15: Explainability & Confidence Reasoning Engine",
            "model_versions": versions,
            "config_hash": f"sha256:{config_hash}",
            "reproducibility": "deterministic",
            "explainability_version": versions.get("explainability_engine", "1.0.0")
        }
