"""
astra_bitstream_transformer
ASTRA Stage 13 — 1D CNN + Transformer Bitstream Structure Model
"""

from .models import (
    StructureLabel,
    RegionPrediction,
    BitstreamStructurePrediction,
    ASTRABitstreamCNNTransformer,
    LABEL_NAMES,
    LABEL_TO_ID,
    ID_TO_LABEL
)
from .cnn_encoder import BitstreamCNNEncoder, Conv1DResidualBlock
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer import BitstreamTransformerEncoder
from .heads import SequenceClassificationHead, BoundaryPredictionHead, FrameClassificationHead
from .dataset import BitstreamStructureDataset, create_synthetic_dataset
from .augmentations import apply_ber_augmentation, apply_offset_augmentation, add_confidence_noise
from .losses import CombinedBitstreamLoss
from .train import run_tiny_overfit_test, train_bitstream_model
from .evaluate import (
    compute_sequence_metrics,
    compute_boundary_f1,
    compute_region_iou,
    evaluate_model
)
from .windowing import slice_sequence_windows
from .merge import merge_overlapping_predictions
from .postprocess import (
    apply_unknown_confidence_threshold,
    extract_contiguous_regions,
    filter_small_regions,
    compute_uncertainty_metrics
)
from .inference import BitstreamStructureModel
from .utils import (
    save_checkpoint,
    load_checkpoint,
    hex_to_bits,
    generate_annotated_synthetic_stream
)

__all__ = [
    "StructureLabel",
    "RegionPrediction",
    "BitstreamStructurePrediction",
    "ASTRABitstreamCNNTransformer",
    "LABEL_NAMES",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "BitstreamCNNEncoder",
    "Conv1DResidualBlock",
    "SinusoidalPositionalEncoding",
    "BitstreamTransformerEncoder",
    "SequenceClassificationHead",
    "BoundaryPredictionHead",
    "FrameClassificationHead",
    "BitstreamStructureDataset",
    "create_synthetic_dataset",
    "apply_ber_augmentation",
    "apply_offset_augmentation",
    "add_confidence_noise",
    "CombinedBitstreamLoss",
    "run_tiny_overfit_test",
    "train_bitstream_model",
    "compute_sequence_metrics",
    "compute_boundary_f1",
    "compute_region_iou",
    "evaluate_model",
    "slice_sequence_windows",
    "merge_overlapping_predictions",
    "apply_unknown_confidence_threshold",
    "extract_contiguous_regions",
    "filter_small_regions",
    "compute_uncertainty_metrics",
    "BitstreamStructureModel",
    "save_checkpoint",
    "load_checkpoint",
    "hex_to_bits",
    "generate_annotated_synthetic_stream"
]
