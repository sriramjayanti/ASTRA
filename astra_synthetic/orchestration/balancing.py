"""
Class Balancing and Parameter Sampling for ASTRA Dataset Orchestration (Engine 8).
Provides balanced round-robin and weighted stochastic selection of modulation,
coding, interleaving, channel, and container profiles while preserving statistical independence.
"""

from __future__ import annotations

from typing import Any
import numpy as np


# Built-in profile mappings
DEFAULT_MODULATION_PROFILES = {
    "2fsk": "2fsk_cp",
    "4fsk": "4fsk_cp",
    "msk": "msk_cp",
    "bpsk": "bpsk_rrc",
    "qpsk": "qpsk_rrc",
    "8psk": "8psk_rrc",
    "16qam": "16qam_rrc",
    "64qam": "64qam_rrc",
    "256qam": "256qam_rrc",
}

DEFAULT_FEC_PROFILES = {
    "none": "none",
    "convolutional": "conv_k7_r12",
    "reed_solomon": "rs_255_223",
    "concatenated": "concat_rs255223_conv_k7",
    "ldpc": "ldpc_n256_k128_r12",
}

DEFAULT_INTERLEAVER_PROFILES = {
    "none": "none",
    "block": "block_16x32",
    "convolutional": "conv_b8_d4",
    "diagonal": "diag_16x16",
    "pseudo_random": "pr_256",
}

DEFAULT_CHANNEL_PROFILES = {
    "clean": "clean",
    "easy": "combined_easy",
    "medium": "combined_medium",
    "hard": "combined_hard",
    "satellite_like": "satellite_like_v1",
}

DEFAULT_CAPTURE_PROFILES = [
    "raw_f32_le_iq",
    "raw_f32_be_iq",
    "raw_i16_le_iq",
    "raw_i16_le_qi",
    "raw_i8_iq",
    "wav_i16_iq",
    "wav_i16_qi",
    "sigmf_cf32_le",
]


class ProfileSampler:
    """Samples valid, statistically balanced pipeline configurations for each synthetic chain."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        dist_cfg = self.config.get("distributions", {})

        self.mod_dist = dist_cfg.get("modulation", {k: 1.0 / len(DEFAULT_MODULATION_PROFILES) for k in DEFAULT_MODULATION_PROFILES})
        self.fec_dist = dist_cfg.get("fec", {k: 1.0 / len(DEFAULT_FEC_PROFILES) for k in DEFAULT_FEC_PROFILES})
        self.int_dist = dist_cfg.get("interleaver", {k: 1.0 / len(DEFAULT_INTERLEAVER_PROFILES) for k in DEFAULT_INTERLEAVER_PROFILES})
        self.chan_dist = dist_cfg.get("channel_difficulty", {k: 1.0 / len(DEFAULT_CHANNEL_PROFILES) for k in DEFAULT_CHANNEL_PROFILES})
        self.cap_dist = dist_cfg.get("capture_profiles", {p: 1.0 / len(DEFAULT_CAPTURE_PROFILES) for p in DEFAULT_CAPTURE_PROFILES})

        # Precompute keys and normalized probabilities
        self._mod_keys, self._mod_probs = self._normalize_dist(self.mod_dist)
        self._fec_keys, self._fec_probs = self._normalize_dist(self.fec_dist)
        self._int_keys, self._int_probs = self._normalize_dist(self.int_dist)
        self._chan_keys, self._chan_probs = self._normalize_dist(self.chan_dist)
        self._cap_keys, self._cap_probs = self._normalize_dist(self.cap_dist)

    @staticmethod
    def _normalize_dist(d: dict[str, float]) -> tuple[list[str], np.ndarray]:
        keys = list(d.keys())
        weights = np.array([float(d[k]) for k in keys], dtype=np.float64)
        total = np.sum(weights)
        if total <= 0:
            weights = np.ones(len(keys), dtype=np.float64)
            total = len(keys)
        probs = weights / total
        return keys, probs

    def sample_chain_parameters(self, rng: np.random.Generator, balanced_index: int | None = None) -> dict[str, Any]:
        """Sample a complete, independent set of parameters for a single pipeline run."""
        if balanced_index is not None:
            # Deterministic round-robin cycling across classes for perfect initial balance
            mod_type = self._mod_keys[balanced_index % len(self._mod_keys)]
            fec_type = self._fec_keys[balanced_index % len(self._fec_keys)]
            int_type = self._int_keys[balanced_index % len(self._int_keys)]
            chan_diff = self._chan_keys[balanced_index % len(self._chan_keys)]
            cap_prof = self._cap_keys[balanced_index % len(self._cap_keys)]
        else:
            mod_type = rng.choice(self._mod_keys, p=self._mod_probs)
            fec_type = rng.choice(self._fec_keys, p=self._fec_probs)
            int_type = rng.choice(self._int_keys, p=self._int_probs)
            chan_diff = rng.choice(self._chan_keys, p=self._chan_probs)
            cap_prof = rng.choice(self._cap_keys, p=self._cap_probs)

        # Draw independent payload configuration
        payload_types = ["random_bits", "counter", "text", "repeated_pattern", "high_entropy"]
        payload_type = str(rng.choice(payload_types))
        payload_bits = int(rng.choice([256, 512, 1024, 2048]))

        # Draw independent symbol rate & sample rate
        symbol_rates = [9600.0, 19200.0, 24000.0, 48000.0]
        symbol_rate = float(rng.choice(symbol_rates))
        samples_per_symbol = int(rng.choice([4, 8, 16]))
        sample_rate = float(symbol_rate * samples_per_symbol)

        return {
            "payload": {
                "payload_type": payload_type,
                "bit_length": payload_bits,
            },
            "framing": {
                "frame_type": "standard",
            },
            "fec": {
                "family": fec_type,
                "profile": DEFAULT_FEC_PROFILES.get(fec_type, "none"),
            },
            "interleaving": {
                "family": int_type,
                "profile": DEFAULT_INTERLEAVER_PROFILES.get(int_type, "none"),
            },
            "modulation": {
                "modulation_type": mod_type,
                "profile": DEFAULT_MODULATION_PROFILES.get(mod_type, "qpsk_rrc"),
                "symbol_rate": symbol_rate,
                "sample_rate": sample_rate,
                "samples_per_symbol": samples_per_symbol,
            },
            "channel": {
                "difficulty": chan_diff,
                "profile": DEFAULT_CHANNEL_PROFILES.get(chan_diff, "combined_medium"),
            },
            "capture": {
                "profile": cap_prof,
            },
        }
