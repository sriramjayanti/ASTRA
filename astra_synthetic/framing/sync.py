"""
Sync Word Generator for ASTRA Framing Engine.
Generates fixed, pool-selected, and random synchronization sequences for frame alignment.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from ..payload.models import hex_to_bits, bits_to_hex


class SyncWordGenerator:
    """Generates synchronization headers according to configured mode and seed."""

    def __init__(self, config: dict[str, Any] | None = None, rng: np.random.Generator | None = None):
        """Initialize SyncWordGenerator with configuration and RNG.

        Args:
            config: Sync configuration dictionary.
            rng: NumPy random generator instance.
        """
        self.config = config or {"mode": "fixed", "format": "hex", "value": "1ACFFC1D"}
        self._rng = rng if rng is not None else np.random.default_rng(42)

    def set_rng(self, rng: np.random.Generator) -> None:
        """Update RNG instance for deterministic generation."""
        self._rng = rng

    def generate(
        self,
        mode: str | None = None,
        value: str | None = None,
        length_bits: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, str, int, str]:
        """Generate sync word bits and metadata.

        Args:
            mode: 'fixed', 'random', or 'pool'. Defaults to config.
            value: Explicit sync pattern string if overriding.
            length_bits: Length in bits (used for random mode).
            seed: Seed for random generator.

        Returns:
            tuple (sync_bits, sync_value_str, sync_length_bits, sync_type_str)
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        sync_mode = mode or self.config.get("mode", "fixed")
        sync_fmt = self.config.get("format", "hex")

        if sync_mode == "fixed":
            val_str = value or self.config.get("value", "1ACFFC1D")
            if not isinstance(val_str, str):
                raise ValueError(f"Sync value must be a string, got {type(val_str).__name__}")
            val_str = val_str.strip()
            
            # Detect format if not specified
            if sync_fmt == "binary" or sync_fmt == "bits" or (set(val_str).issubset({"0", "1"}) and len(val_str) > 8 and sync_fmt != "hex"):
                # Binary string
                if not set(val_str).issubset({"0", "1"}):
                    raise ValueError(f"Invalid binary sync pattern '{val_str}'")
                bits = np.array([int(c) for c in val_str], dtype=np.uint8)
                sync_val_repr = val_str
            else:
                # Hex string
                bits = hex_to_bits(val_str)
                sync_val_repr = val_str.upper().replace("0X", "").strip()

            sync_len = len(bits)
            return bits, sync_val_repr, sync_len, "fixed"

        elif sync_mode == "pool":
            pool_values: Sequence[str] = self.config.get(
                "values", ["1ACFFC1D", "D391D391", "A5A5A5A5"]
            )
            if not pool_values:
                raise ValueError("Sync mode 'pool' requires non-empty 'values' list in config")
            
            idx = int(rng.integers(0, len(pool_values)))
            chosen_val = pool_values[idx].strip()
            
            if sync_fmt == "binary" or sync_fmt == "bits":
                if not set(chosen_val).issubset({"0", "1"}):
                    raise ValueError(f"Invalid binary sync pattern in pool '{chosen_val}'")
                bits = np.array([int(c) for c in chosen_val], dtype=np.uint8)
                sync_val_repr = chosen_val
            else:
                bits = hex_to_bits(chosen_val)
                sync_val_repr = chosen_val.upper().replace("0X", "").strip()

            sync_len = len(bits)
            return bits, sync_val_repr, sync_len, "pool"

        elif sync_mode == "random":
            len_b = length_bits or self.config.get("length_bits", 32)
            if len_b < 1:
                raise ValueError(f"Sync length_bits must be >= 1, got {len_b}")
            bits = rng.integers(0, 2, size=len_b, dtype=np.uint8)
            sync_val_repr = bits_to_hex(bits, padding=True) if len_b % 8 == 0 else "".join(str(b) for b in bits)
            return bits, sync_val_repr, len_b, "random"

        else:
            raise ValueError(f"Unknown sync mode '{sync_mode}'. Supported modes: 'fixed', 'pool', 'random'")
