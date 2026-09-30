"""
feature_builder.py
Structural feature extraction and multi-channel sequence map generation for Stage 13 CNN + Transformer.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from .models import (
    BitBalance,
    RunLengthStats,
    ByteAlignmentResult,
    AutocorrPeak,
    PeriodicityCandidate,
    SyncPatternResult,
    RepeatedPattern,
    StructuralRegion
)
from .autocorrelation import bits_to_bipolar


FEATURE_SCHEMA_VERSION = "bitstream_features_v1"


class BitstreamFeatureBuilder:
    """
    Builds structured 1D global feature vectors and 2D multi-channel sequence feature maps
    for neural sequence modeling in Stage 13 (CNN + Transformer).
    """

    def __init__(self, schema_version: str = FEATURE_SCHEMA_VERSION):
        self.schema_version = schema_version

    def build_global_features(
        self,
        bit_count: int,
        bit_balance: BitBalance,
        binary_entropy_global: float,
        ngram_entropies: Dict[str, float],
        run_length_stats: RunLengthStats,
        autocorr_peaks: List[AutocorrPeak],
        periodicity_candidates: List[PeriodicityCandidate],
        frame_length_candidates: List[PeriodicityCandidate],
        repeated_patterns: List[RepeatedPattern],
        byte_alignment: ByteAlignmentResult
    ) -> Dict[str, float]:
        """Generate compact 1D scalar feature vector."""
        max_peak = autocorr_peaks[0].correlation if autocorr_peaks else 0.0
        max_lag = float(autocorr_peaks[0].lag) if autocorr_peaks else 0.0

        top_period_score = periodicity_candidates[0].score if periodicity_candidates else 0.0
        top_frame_len = float(frame_length_candidates[0].period_bits) if frame_length_candidates else 0.0
        top_frame_score = frame_length_candidates[0].score if frame_length_candidates else 0.0

        margin = 0.0
        if len(frame_length_candidates) >= 2:
            margin = frame_length_candidates[0].score - frame_length_candidates[1].score
        elif len(frame_length_candidates) == 1:
            margin = frame_length_candidates[0].score

        best_byte_score = byte_alignment.structural_scores[byte_alignment.best_offset] if byte_alignment.structural_scores else 0.0

        return {
            "bit_count": float(bit_count),
            "bit_balance_imbalance": float(bit_balance.imbalance),
            "binary_entropy_global": float(binary_entropy_global),
            "ngram_entropy_2bit": float(ngram_entropies.get("2bit", binary_entropy_global)),
            "ngram_entropy_4bit": float(ngram_entropies.get("4bit", binary_entropy_global)),
            "ngram_entropy_8bit": float(ngram_entropies.get("8bit", binary_entropy_global)),
            "max_autocorr_peak": float(max_peak),
            "max_autocorr_lag": float(max_lag),
            "periodicity_strength": float(top_period_score),
            "top_frame_length": float(top_frame_len),
            "top_frame_score": float(top_frame_score),
            "frame_length_margin": float(margin),
            "repeat_pattern_count": float(len(repeated_patterns)),
            "mean_run_length": float(run_length_stats.mean_run_length),
            "max_run_length": float(run_length_stats.max_run_length),
            "run_length_variance": float(run_length_stats.run_length_variance),
            "long_run_fraction": float(run_length_stats.long_run_fraction),
            "best_byte_offset": float(byte_alignment.best_offset),
            "best_byte_score": float(best_byte_score),
        }

    def build_sequence_feature_map(
        self,
        bits: np.ndarray,
        soft_info: Optional[np.ndarray] = None,
        entropy_windows: Optional[List[float]] = None,
        window_step: int = 16,
        top_frame_length: Optional[int] = None,
        position_stability: Optional[np.ndarray] = None,
        sync_positions: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Construct multi-channel 2D sequence feature array of shape [N_bits, 4].

        Channels:
          0: Bipolar bit value in {-1.0, +1.0}
          1: Confidence / soft reliability (defaults to 1.0)
          2: Local sliding-window entropy interpolated per bit
          3: Positional stability / frame boundary marker
        """
        n = len(bits)
        channels = np.zeros((n, 4), dtype=np.float32)

        # Channel 0: Bipolar bits
        channels[:, 0] = bits_to_bipolar(bits)

        # Channel 1: Soft info / confidence
        if soft_info is not None and len(soft_info) == n:
            channels[:, 1] = np.abs(soft_info) / max(1.0, float(np.max(np.abs(soft_info))))
        else:
            channels[:, 1] = 1.0

        # Channel 2: Interpolated sliding-window entropy
        if entropy_windows and len(entropy_windows) > 1:
            x_win = np.arange(len(entropy_windows)) * window_step
            x_bits = np.arange(n)
            # Linear interpolation of window entropy onto each bit position
            interp_ent = np.interp(x_bits, x_win, entropy_windows, left=entropy_windows[0], right=entropy_windows[-1])
            channels[:, 2] = interp_ent.astype(np.float32)
        else:
            channels[:, 2] = 1.0

        # Channel 3: Positional stability / Frame periodic marker
        if top_frame_length is not None and top_frame_length > 0:
            if position_stability is not None and len(position_stability) == top_frame_length:
                # Tile positional stability across bitstream
                tiled_stab = np.tile(position_stability, int(np.ceil(n / top_frame_length)))[:n]
                channels[:, 3] = tiled_stab.astype(np.float32)
            else:
                # Modulo ramp marker
                channels[:, 3] = (np.arange(n) % top_frame_length) / float(top_frame_length)

        # Mark confirmed sync positions with impulse
        if sync_positions:
            for p in sync_positions:
                if 0 <= p < n:
                    channels[p, 3] = 1.0

        return channels
