"""
Deterministic Hierarchical Seed Manager for ASTRA Dataset Orchestration (Engine 8).
Uses NumPy SeedSequence to generate reproducible, statistically independent seeds per stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ChainSeeds:
    """Independent seeds derived for each stage in a single source chain."""
    master_chain_seed: int
    payload_seed: int
    frame_seed: int
    fec_seed: int
    interleaver_seed: int
    modulation_seed: int
    channel_seed: int
    capture_seed: int

    def to_dict(self) -> dict[str, int]:
        return {
            "master_chain_seed": self.master_chain_seed,
            "payload_seed": self.payload_seed,
            "frame_seed": self.frame_seed,
            "fec_seed": self.fec_seed,
            "interleaver_seed": self.interleaver_seed,
            "modulation_seed": self.modulation_seed,
            "channel_seed": self.channel_seed,
            "capture_seed": self.capture_seed,
        }


class SeedManager:
    """Manages hierarchical deterministic seed derivation for the ASTRA synthetic pipeline."""

    def __init__(self, master_seed: int = 42):
        self.master_seed = int(master_seed)
        self._master_seq = np.random.SeedSequence(self.master_seed)

    def get_chain_seeds(self, chain_index: int) -> ChainSeeds:
        """Derive independent seeds for a specific chain index."""
        # Spawn child sequence for this specific chain index
        chain_seq = self._master_seq.spawn(chain_index + 1)[0]
        # Spawn 8 child sequences for the 7 stages + master
        stage_seqs = chain_seq.spawn(8)

        # Generate 32-bit positive integer seeds
        seeds = [int(s.generate_state(1)[0] & 0x7FFFFFFF) for s in stage_seqs]

        return ChainSeeds(
            master_chain_seed=seeds[0],
            payload_seed=seeds[1],
            frame_seed=seeds[2],
            fec_seed=seeds[3],
            interleaver_seed=seeds[4],
            modulation_seed=seeds[5],
            channel_seed=seeds[6],
            capture_seed=seeds[7],
        )
