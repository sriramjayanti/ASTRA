"""
ASTRA Multi-Branch Modulation Fusion Package.
"""

from .models import (
    BranchPrediction,
    CandidateItem,
    ClassMappingMismatchError,
    FusionPrediction,
    InvalidProbabilityError,
    InvalidWeightError,
    SourceAlignmentError,
)
from .adapters import (
    BaseBranchAdapter,
    ResNet1DAdapter,
    Spectrogram2DAdapter,
    preprocess_iq,
)
from .weighted_fusion import WeightedProbabilityFusion
from .learned_fusion import (
    FusionFeatureDataset,
    LearnedFusionEngine,
    LearnedFusionMLP,
)
from .confidence import ConfidenceCalculator
from .calibration import TemperatureScaler
from .inference import ASTRAFusionEngine
from .checkpoint import load_fusion_checkpoint, save_fusion_checkpoint
from .metrics import (
    compute_agreement_metrics,
    compute_comprehensive_metrics,
    compute_expected_calibration_error,
    grid_search_fusion_weights,
)
from .utils import (
    BenchmarkTimer,
    format_fusion_prediction_summary,
    setup_logger,
)

__version__ = "1.0.0"

__all__ = [
    "ASTRAFusionEngine",
    "BaseBranchAdapter",
    "ResNet1DAdapter",
    "Spectrogram2DAdapter",
    "BranchPrediction",
    "CandidateItem",
    "FusionPrediction",
    "WeightedProbabilityFusion",
    "LearnedFusionEngine",
    "LearnedFusionMLP",
    "FusionFeatureDataset",
    "ConfidenceCalculator",
    "TemperatureScaler",
    "ClassMappingMismatchError",
    "SourceAlignmentError",
    "InvalidProbabilityError",
    "InvalidWeightError",
    "preprocess_iq",
    "load_fusion_checkpoint",
    "save_fusion_checkpoint",
    "compute_agreement_metrics",
    "compute_comprehensive_metrics",
    "compute_expected_calibration_error",
    "grid_search_fusion_weights",
    "format_fusion_prediction_summary",
    "setup_logger",
    "BenchmarkTimer",
]
