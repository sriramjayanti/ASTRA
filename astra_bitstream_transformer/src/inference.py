"""
inference.py
High-level prediction orchestrator for Stage 13 1D CNN + Transformer Bitstream Structure Model.
"""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import os
import yaml
import numpy as np
import torch

from .models import (
    StructureLabel,
    RegionPrediction,
    BitstreamStructurePrediction,
    ASTRABitstreamCNNTransformer,
    LABEL_NAMES
)
from .windowing import slice_sequence_windows
from .merge import merge_overlapping_predictions
from .postprocess import (
    apply_unknown_confidence_threshold,
    extract_contiguous_regions,
    filter_small_regions,
    compute_uncertainty_metrics
)
from .utils import load_checkpoint


class BitstreamStructureModel:
    """
    Production-ready Bitstream Structure Prediction Model (ASTRA Stage 13).
    Receives recovered candidate bitstreams and Stage 12 side-information, predicting
    bit-level and region-level classifications (SYNC, HEADER, PAYLOAD, CRC, PADDING, UNKNOWN).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.config = self._load_config(config_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        m_conf = self.config.get("model", {})
        cnn_conf = m_conf.get("cnn", {})
        trans_conf = m_conf.get("transformer", {})

        self.model = ASTRABitstreamCNNTransformer(
            in_channels=m_conf.get("input_channels", 6),
            cnn_channels=cnn_conf.get("channels", [64, 128, 256]),
            cnn_kernel_size=cnn_conf.get("kernel_size", 7),
            cnn_residual_blocks=cnn_conf.get("residual_blocks_per_stage", 2),
            d_model=trans_conf.get("d_model", 256),
            num_heads=trans_conf.get("num_heads", 8),
            num_layers=trans_conf.get("num_layers", 4),
            dim_feedforward=trans_conf.get("dim_feedforward", 1024),
            num_classes=m_conf.get("output", {}).get("num_classes", 6),
            enable_frame_head=m_conf.get("output", {}).get("enable_frame_head", True),
            dropout=trans_conf.get("dropout", 0.1)
        ).to(self.device)

        self.model.eval()

        if weights_path and os.path.exists(weights_path):
            load_checkpoint(weights_path, self.model, device=self.device)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_config = {
            "model": {
                "version": "astra_bitstream_cnn_transformer_v1",
                "input_schema_version": "bitstream_model_input_v1",
                "input_channels": 6,
                "cnn": {"channels": [64, 128, 256], "kernel_size": 7, "residual_blocks_per_stage": 2},
                "transformer": {"d_model": 256, "num_heads": 8, "num_layers": 4, "dim_feedforward": 1024, "dropout": 0.1},
                "output": {"num_classes": 6, "enable_frame_head": True, "frame_classes": 4}
            },
            "training": {"window_length": 2048, "overlap_fraction": 0.5},
            "postprocessing": {
                "unknown_confidence_threshold": 0.40,
                "min_region_sizes": {"UNKNOWN": 4, "SYNC": 8, "HEADER": 8, "PAYLOAD": 16, "CRC": 4, "PADDING": 4}
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    return loaded

        pkg_conf = Path(__file__).parent.parent / "configs" / "model_config.yaml"
        if pkg_conf.exists():
            with open(pkg_conf, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    return loaded

        return default_config

    def _prepare_channel_tensor(
        self,
        bits: np.ndarray,
        stage12_result: Optional[Any] = None,
        soft_info: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Assemble the 6-channel input sequence tensor [6, N] from raw bits and Stage 12 structural outputs.
        """
        N = len(bits)
        channels = np.zeros((6, N), dtype=np.float32)

        # Ch 0: Bipolar bits
        channels[0, :] = 2.0 * bits.astype(np.float32) - 1.0

        # Ch 1: Soft info / reliability
        if soft_info is not None and len(soft_info) == N:
            max_s = max(1.0, float(np.max(np.abs(soft_info))))
            channels[1, :] = np.abs(soft_info) / max_s
        elif stage12_result is not None and hasattr(stage12_result, "sequence_feature_map") and stage12_result.sequence_feature_map is not None:
            channels[1, :] = stage12_result.sequence_feature_map[:N, 1]
        else:
            channels[1, :] = 1.0

        # Ch 2: Local sliding entropy
        if stage12_result is not None and hasattr(stage12_result, "entropy_windows") and stage12_result.entropy_windows:
            ew = stage12_result.entropy_windows
            x_win = np.linspace(0, N - 1, len(ew))
            channels[2, :] = np.interp(np.arange(N), x_win, ew).astype(np.float32)
        else:
            # Approximate sliding entropy
            win_w = 64
            for i in range(0, N, 32):
                w = bits[max(0, i - win_w // 2):min(N, i + win_w // 2)]
                p1 = float(np.mean(w))
                p0 = 1.0 - p1
                ent = - (p0 * np.log2(p0) + p1 * np.log2(p1)) if (p0 > 1e-9 and p1 > 1e-9) else 0.0
                channels[2, i:min(N, i + 32)] = float(ent)

        # Ch 3: Frame boundary / periodic ramp
        top_fl = None
        if stage12_result is not None and hasattr(stage12_result, "frame_length_candidates") and stage12_result.frame_length_candidates:
            top_fl = stage12_result.frame_length_candidates[0].period_bits

        if top_fl and top_fl > 0:
            channels[3, :] = (np.arange(N) % top_fl) / float(top_fl)
        else:
            channels[3, :] = (np.arange(N) % 512) / 512.0

        # Ch 4: Sync candidate evidence
        sync_positions = []
        if stage12_result is not None:
            if hasattr(stage12_result, "sync_results") and stage12_result.sync_results:
                sync_positions = stage12_result.sync_results[0].positions
            elif hasattr(stage12_result, "candidate_sync_words") and stage12_result.candidate_sync_words:
                sync_positions = stage12_result.candidate_sync_words[0].positions

        for p in sync_positions:
            if 0 <= p < N:
                channels[4, p:min(N, p + 16)] = 1.0

        # Ch 5: Positional stability map
        if stage12_result is not None and hasattr(stage12_result, "position_stability_map") and stage12_result.position_stability_map:
            stab = np.array(stage12_result.position_stability_map, dtype=np.float32)
            if len(stab) > 0:
                channels[5, :] = np.tile(stab, int(np.ceil(N / len(stab))))[:N]

        return channels

    def predict(
        self,
        bits: np.ndarray,
        stage12_result: Optional[Any] = None,
        soft_info: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> BitstreamStructurePrediction:
        """
        Predict bitstream structure on input bits.
        """
        context = context or {}
        pipeline_path_id = str(context.get("pipeline_path_id", "candidate_0"))

        if not isinstance(bits, np.ndarray):
            bits = np.asarray(bits)
        if bits.ndim > 1:
            bits = bits.ravel()

        N = len(bits)
        if N == 0:
            return BitstreamStructurePrediction(
                pipeline_path_id=pipeline_path_id,
                sequence_length=0,
                predicted_labels=np.array([], dtype=np.int64),
                label_probabilities=np.zeros((0, 6), dtype=np.float32)
            )

        channels = self._prepare_channel_tensor(bits, stage12_result=stage12_result, soft_info=soft_info)

        win_len = self.config.get("training", {}).get("window_length", 2048)
        overlap = self.config.get("training", {}).get("overlap_fraction", 0.5)

        windows, masks, _, slices = slice_sequence_windows(
            channels=channels,
            window_length=win_len,
            overlap_fraction=overlap
        )

        tensor_wins = torch.from_numpy(windows).to(self.device)
        tensor_masks = torch.from_numpy(masks).to(self.device)

        window_probs_list = []
        frame_preds = {}

        self.model.eval()
        with torch.no_grad():
            for i in range(len(windows)):
                w_x = tensor_wins[i:i + 1]
                w_m = tensor_masks[i:i + 1]
                out = self.model(w_x, padding_mask=w_m)

                logits = out["sequence_logits"][0] # [L, num_classes]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                window_probs_list.append(probs)

                if "frame_logits" in out and out["frame_logits"] is not None:
                    f_logits = out["frame_logits"][0].cpu().numpy()
                    f_probs = 1.0 / (1.0 + np.exp(-f_logits))
                    frame_preds = {
                        "valid_structure": float(f_probs[0]) if len(f_probs) > 0 else 0.0,
                        "header_present": float(f_probs[1]) if len(f_probs) > 1 else 0.0,
                        "payload_present": float(f_probs[2]) if len(f_probs) > 2 else 0.0,
                        "crc_present": float(f_probs[3]) if len(f_probs) > 3 else 0.0,
                    }

        window_probs_arr = np.stack(window_probs_list, axis=0)

        # Merge overlapping predictions
        full_probs, raw_labels = merge_overlapping_predictions(
            window_probs=window_probs_arr,
            window_slices=slices,
            total_length=N,
            num_classes=self.config.get("model", {}).get("output", {}).get("num_classes", 6),
            window_length=win_len
        )

        # Postprocessing: Unknown threshold and Contiguous Region extraction
        unk_thresh = self.config.get("postprocessing", {}).get("unknown_confidence_threshold", 0.40)
        refined_labels = apply_unknown_confidence_threshold(
            raw_labels,
            full_probs,
            unknown_threshold=unk_thresh
        )

        raw_regions = extract_contiguous_regions(refined_labels, full_probs)
        min_sizes = self.config.get("postprocessing", {}).get("min_region_sizes")
        regions = filter_small_regions(raw_regions, min_region_sizes=min_sizes)

        # Boundary positions (indices where region changes)
        boundaries = [r.start_bit for r in regions[1:]]

        # Uncertainty metrics
        mean_uncertainty, uncertain_spans = compute_uncertainty_metrics(full_probs)
        model_conf = float(np.mean(np.max(full_probs, axis=-1)))

        return BitstreamStructurePrediction(
            pipeline_path_id=pipeline_path_id,
            sequence_length=N,
            predicted_labels=refined_labels,
            label_probabilities=full_probs,
            regions=regions,
            boundary_positions=boundaries,
            frame_level_predictions=frame_preds,
            model_confidence=model_conf,
            mean_uncertainty=mean_uncertainty,
            uncertain_regions=uncertain_spans,
            model_version=self.config.get("model", {}).get("version", "astra_bitstream_cnn_transformer_v1"),
            feature_schema_version=self.config.get("model", {}).get("input_schema_version", "bitstream_model_input_v1")
        )

    def predict_batch(
        self,
        batch_bits: List[np.ndarray],
        stage12_results: Optional[List[Any]] = None,
        context_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[BitstreamStructurePrediction]:
        """Run predictions on batch of bitstreams."""
        results = []
        for i, bits in enumerate(batch_bits):
            s12 = stage12_results[i] if stage12_results and i < len(stage12_results) else None
            ctx = context_list[i] if context_list and i < len(context_list) else {}
            results.append(self.predict(bits, stage12_result=s12, context=ctx))
        return results
