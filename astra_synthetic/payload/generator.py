"""
Production-quality Payload / Bitstream Generator for ASTRA.
Generates controlled, deterministic ground-truth payloads for synthetic dataset generation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Generator, Iterator, Sequence
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .models import (
    PayloadRecord,
    bytes_to_bits,
    bits_to_bytes,
    hex_to_bits,
    text_to_bits,
    calculate_entropy,
)
from .validators import (
    ValidationError,
    validate_pattern,
    validate_probability,
    validate_hex_string,
    validate_payload_record,
    validate_config,
)

logger = logging.getLogger("astra_synthetic.payload")


DEFAULT_CONFIG: dict[str, Any] = {
    "payload_generator": {
        "version": "1.0.0",
        "master_seed": 42,
        "bit_order": "msb_first",
        "length": {
            "mode": "random",
            "min_bits": 256,
            "max_bits": 4096,
        },
        "types": {
            "random_bits": {
                "enabled": True,
                "probability": 0.40,
            },
            "random_bytes": {
                "enabled": True,
                "probability": 0.20,
            },
            "repeated_pattern": {
                "enabled": True,
                "probability": 0.10,
                "patterns": ["01", "1010", "11110000", "10110011"],
            },
            "biased_random": {
                "enabled": True,
                "probability": 0.10,
                "probability_one_range": {"min": 0.02, "max": 0.20},
            },
            "counter": {
                "enabled": True,
                "probability": 0.10,
            },
            "text": {
                "enabled": True,
                "probability": 0.05,
                "corpus": [
                    "ASTRA SIGNAL TEST FRAME 001",
                    "SYNTHETIC BITSTREAM TRANSMISSION",
                    "GROUND TRUTH TELEMETRY PACKET",
                    "AUTOMATED SIGNAL RECOVERY 2026",
                ],
            },
            "fixed_hex": {
                "enabled": True,
                "probability": 0.05,
                "values": ["DEADBEEF", "CAFEBABE", "123456789ABCDEF0"],
            },
        },
    }
}


class PayloadGenerator:
    """Deterministic Payload / Bitstream Generator for ASTRA synthetic pipeline."""

    VERSION = "1.0.0"

    def __init__(self, config: dict[str, Any] | str | Path | None = None):
        """Initialize PayloadGenerator with an optional configuration dictionary or file path.

        Args:
            config: Config dict, YAML file path, or None (uses default configuration).
        """
        self.raw_config = self._load_config(config)
        validate_config(self.raw_config)

        pg_cfg = self.raw_config.get("payload_generator", self.raw_config)
        self.version = pg_cfg.get("version", self.VERSION)
        self.master_seed = pg_cfg.get("master_seed", 42)
        self.bit_order = pg_cfg.get("bit_order", "msb_first")
        
        # Length config
        self.length_cfg = pg_cfg.get("length", {"mode": "random", "min_bits": 256, "max_bits": 4096})
        
        # Types config
        self.types_cfg = pg_cfg.get("types", {})
        
        # Initialize master RNG
        self._rng = np.random.default_rng(self.master_seed)
        self._payload_counter = 0

    def _load_config(self, config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
        """Load configuration from dictionary, YAML file, or default."""
        if config is None:
            return DEFAULT_CONFIG.copy()
        
        if isinstance(config, dict):
            return config
        
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            if HAS_YAML and (path.suffix.lower() in [".yaml", ".yml"]):
                loaded = yaml.safe_load(f)
            else:
                # Basic json or fallback
                import json
                try:
                    loaded = json.load(f)
                except Exception:
                    if not HAS_YAML:
                        raise ImportError(
                            "PyYAML is required to parse .yaml files. Please install pyyaml or pass a dict."
                        )
                    raise
            return loaded if isinstance(loaded, dict) else {}

    def _next_payload_id(self) -> str:
        """Generate next sequential payload ID."""
        self._payload_counter += 1
        return f"payload_{self._payload_counter:06d}"

    def reset(self, seed: int | None = None) -> None:
        """Reset internal generator state and ID counter.

        Args:
            seed: Optional new seed. If None, resets to master_seed.
        """
        if seed is not None:
            self.master_seed = seed
        self._rng = np.random.default_rng(self.master_seed)
        self._payload_counter = 0

    def generate(
        self,
        payload_type: str = "random_bits",
        bit_length: int | None = None,
        byte_length: int | None = None,
        seed: int | None = None,
        payload_id: str | None = None,
        **kwargs: Any,
    ) -> PayloadRecord:
        """Generate a single PayloadRecord of the requested type and parameters.

        Args:
            payload_type: One of 'random_bits', 'random_bytes', 'text', 'repeated_pattern',
                          'counter', 'biased_random', 'fixed_hex', 'fixed_bits', etc.
            bit_length: Desired length in bits (takes precedence over random length).
            byte_length: Desired length in bytes (converted to bit_length = byte_length * 8).
            seed: Explicit random seed for deterministic generation.
            payload_id: Explicit payload ID. If None, uses sequential generator.
            **kwargs: Type-specific arguments (e.g. pattern, text, probability_one, value, etc.).

        Returns:
            Validated PayloadRecord.
        """
        pid = payload_id if payload_id is not None else self._next_payload_id()

        # Handle explicit seed or derive a deterministic child seed from master rng
        effective_seed = seed
        if effective_seed is None and payload_type in ("random_bits", "random_bytes", "biased_random"):
            effective_seed = int(self._rng.integers(0, 2**31 - 1))

        rng = np.random.default_rng(effective_seed) if effective_seed is not None else self._rng

        # Determine bit_length
        if bit_length is None:
            if byte_length is not None:
                bit_length = byte_length * 8
            else:
                # Determine from kwargs, config, or payload_type
                if payload_type == "random_bytes":
                    b_len = kwargs.get("byte_length", 128)
                    bit_length = b_len * 8
                elif payload_type == "text":
                    # bit_length derived directly from text encoding
                    pass
                elif payload_type == "counter":
                    b_len = kwargs.get("byte_length", 256)
                    bit_length = b_len * 8
                elif payload_type in ("fixed_hex", "fixed_bits"):
                    # bit_length derived from value
                    pass
                else:
                    # Sample length from config
                    min_b = self.length_cfg.get("min_bits", 256)
                    max_b = self.length_cfg.get("max_bits", 4096)
                    if self.length_cfg.get("mode") == "fixed":
                        bit_length = min_b
                    else:
                        bit_length = int(rng.integers(min_b, max_b + 1))

        # Dispatch generation based on type
        source_text: str | None = None
        pattern: str | None = None
        metadata: dict[str, Any] = {
            "bit_order": self.bit_order,
            "generator_version": self.version,
        }

        if payload_type == "random_bits":
            if bit_length is None:
                bit_length = 1024
            if bit_length < 1:
                raise ValidationError(f"bit_length must be >= 1, got {bit_length}")
            bits = rng.integers(0, 2, size=bit_length, dtype=np.uint8)
            raw_bytes = bits_to_bytes(bits, padding=True)

        elif payload_type == "random_bytes":
            if byte_length is None:
                byte_length = (bit_length + 7) // 8 if bit_length else 128
            if byte_length < 1:
                raise ValidationError(f"byte_length must be >= 1, got {byte_length}")
            byte_arr = rng.integers(0, 256, size=byte_length, dtype=np.uint8)
            raw_bytes = byte_arr.tobytes()
            bits = bytes_to_bits(raw_bytes)
            if bit_length and len(bits) > bit_length:
                bits = bits[:bit_length]

        elif payload_type == "text":
            text_val = kwargs.get("text")
            if text_val is None:
                corpus = self.types_cfg.get("text", {}).get("corpus", ["ASTRA SIGNAL TEST FRAME 001"])
                text_val = corpus[int(rng.integers(0, len(corpus)))]
            if not isinstance(text_val, str):
                raise ValidationError(f"text payload requires string, got {type(text_val).__name__}")
            encoding = kwargs.get("encoding", "utf-8")
            bits = text_to_bits(text_val, encoding=encoding)
            raw_bytes = bits_to_bytes(bits, padding=True)
            source_text = text_val
            metadata["encoding"] = encoding

        elif payload_type == "repeated_pattern":
            pat_val = kwargs.get("pattern")
            if pat_val is None:
                patterns = self.types_cfg.get("repeated_pattern", {}).get(
                    "patterns", ["10110011", "01", "1010"]
                )
                pat_val = patterns[int(rng.integers(0, len(patterns)))]
            validate_pattern(pat_val)
            pattern = pat_val
            
            if bit_length is None:
                bit_length = 2048
            if bit_length < 1:
                raise ValidationError(f"bit_length must be >= 1, got {bit_length}")
                
            pat_bits = np.array([int(c) for c in pat_val], dtype=np.uint8)
            reps = math.ceil(bit_length / len(pat_bits))
            bits = np.tile(pat_bits, reps)[:bit_length]
            raw_bytes = bits_to_bytes(bits, padding=True)
            metadata["pattern"] = pattern

        elif payload_type == "counter":
            start_val = kwargs.get("start_value", 0)
            if not isinstance(start_val, int):
                raise ValidationError(f"start_value must be int, got {type(start_val).__name__}")
            if byte_length is None:
                byte_length = (bit_length + 7) // 8 if bit_length else 256
            if byte_length < 1:
                raise ValidationError(f"byte_length must be >= 1, got {byte_length}")
            
            byte_seq = bytearray((start_val + i) % 256 for i in range(byte_length))
            raw_bytes = bytes(byte_seq)
            bits = bytes_to_bits(raw_bytes)
            if bit_length and len(bits) > bit_length:
                bits = bits[:bit_length]
            metadata["start_value"] = start_val

        elif payload_type in ("biased_random", "low_entropy", "high_entropy"):
            prob_one = kwargs.get("probability_one")
            if prob_one is None:
                if payload_type == "low_entropy":
                    prob_one = 0.05
                elif payload_type == "high_entropy":
                    prob_one = 0.50
                else:
                    range_cfg = self.types_cfg.get("biased_random", {}).get(
                        "probability_one_range", {"min": 0.02, "max": 0.20}
                    )
                    prob_one = float(rng.uniform(range_cfg.get("min", 0.02), range_cfg.get("max", 0.20)))
            validate_probability(prob_one, "probability_one")
            
            if bit_length is None:
                bit_length = 1024
            if bit_length < 1:
                raise ValidationError(f"bit_length must be >= 1, got {bit_length}")
            
            random_floats = rng.random(size=bit_length)
            bits = (random_floats < prob_one).astype(np.uint8)
            raw_bytes = bits_to_bytes(bits, padding=True)
            metadata["probability_one"] = float(round(prob_one, 4))

        elif payload_type == "fixed_hex":
            hex_val = kwargs.get("value")
            if hex_val is None:
                vals = self.types_cfg.get("fixed_hex", {}).get("values", ["DEADBEEF"])
                hex_val = vals[int(rng.integers(0, len(vals)))]
            validate_hex_string(hex_val)
            bits = hex_to_bits(hex_val)
            raw_bytes = bits_to_bytes(bits, padding=True)
            metadata["hex_value"] = hex_val

        elif payload_type == "fixed_bits":
            bit_str = kwargs.get("value")
            if bit_str is None:
                raise ValidationError("fixed_bits requires 'value' argument with binary string")
            validate_pattern(bit_str)
            bits = np.array([int(c) for c in bit_str], dtype=np.uint8)
            raw_bytes = bits_to_bytes(bits, padding=True)
            metadata["bit_value"] = bit_str

        else:
            raise ValidationError(
                f"Unknown payload_type: '{payload_type}'. Supported types: "
                "random_bits, random_bytes, text, repeated_pattern, counter, biased_random, fixed_hex, fixed_bits"
            )

        # Calculate metrics
        actual_bit_len = len(bits)
        actual_byte_len = len(raw_bytes)
        entropy = calculate_entropy(bits)

        record = PayloadRecord(
            payload_id=pid,
            payload_type=payload_type,
            payload_bits=bits,
            payload_bytes=raw_bytes,
            bit_length=actual_bit_len,
            byte_length=actual_byte_len,
            seed=effective_seed,
            source_text=source_text,
            pattern=pattern,
            entropy_estimate=entropy,
            metadata=metadata,
        )

        validate_payload_record(record)
        logger.info("Generated payload %s", record.payload_id)
        logger.debug("type=%s bits=%d seed=%s entropy=%.4f", record.payload_type, record.bit_length, record.seed, record.entropy_estimate)

        return record

    def iter_payloads(
        self,
        count: int,
        config: dict[str, Any] | None = None,
    ) -> Generator[PayloadRecord, None, None]:
        """Stream/yield generated payloads one by one for memory-efficient batch pipelines.

        Args:
            count: Number of payloads to yield.
            config: Optional override configuration for batch mix.

        Yields:
            PayloadRecord instances.
        """
        if count < 1:
            return

        cfg = config if config is not None else self.raw_config
        pg_cfg = cfg.get("payload_generator", cfg)
        types_cfg = pg_cfg.get("types", self.types_cfg)

        # Prepare active types and probability distribution
        active_types: list[str] = []
        probabilities: list[float] = []

        for ptype, pinfo in types_cfg.items():
            if isinstance(pinfo, dict) and pinfo.get("enabled", True):
                active_types.append(ptype)
                probabilities.append(float(pinfo.get("probability", 1.0)))

        if not active_types:
            active_types = ["random_bits"]
            probabilities = [1.0]

        # Normalize probabilities
        prob_arr = np.array(probabilities, dtype=np.float64)
        prob_arr = prob_arr / prob_arr.sum()

        for _ in range(count):
            # Select payload type
            chosen_type = str(self._rng.choice(active_types, p=prob_arr))
            yield self.generate(payload_type=chosen_type)

    def generate_batch(
        self,
        count: int,
        config: dict[str, Any] | None = None,
    ) -> list[PayloadRecord]:
        """Generate a batch of payloads into a list.

        Args:
            count: Number of payloads to generate.
            config: Optional override configuration.

        Returns:
            List of generated PayloadRecord objects.
        """
        return list(self.iter_payloads(count=count, config=config))
