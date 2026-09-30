"""
ASTRA Stage 11 — Pipeline Scoring Model package.
"""

from .models import (
    ConfidenceTier,
    PipelinePathCandidate,
    RankedCandidate,
    PipelineRankingResult,
    EvaluationReport,
)
from .feature_schema import (
    PIPELINE_FEATURE_COLUMNS,
    FEATURE_DEFAULTS,
    PIPELINE_FEATURE_SCHEMA_VERSION,
    get_feature_schema,
)
from .categorical import (
    encode_modulation,
    encode_interleaver_family,
    encode_fec_family,
    encode_validation_status,
)
from .feature_builder import PipelineFeatureBuilder
from .label_builder import evaluate_candidate_correctness
from .dataset_builder import PipelineDatasetBuilder, split_signals_by_group
from .calibration import ScoreCalibrator
from .feature_importance import (
    extract_xgboost_feature_importance,
    explain_candidate_prediction,
)
from .evaluation import (
    evaluate_signal_level_ranking,
    evaluate_pipeline_model,
    compute_expected_calibration_error,
)
from .training import train_pipeline_scorer
from .ranking import (
    rank_pipeline_candidates,
    compute_ranking_uncertainty,
    determine_confidence_tier,
)
from .scorer import PipelineScorer
from .checkpoint import (
    save_pipeline_model_package,
    load_pipeline_model_package,
)
from .inference import PipelineScoringEngine
from .utils import generate_synthetic_candidate_tree

__all__ = [
    "ConfidenceTier",
    "PipelinePathCandidate",
    "RankedCandidate",
    "PipelineRankingResult",
    "EvaluationReport",
    "PIPELINE_FEATURE_COLUMNS",
    "FEATURE_DEFAULTS",
    "PIPELINE_FEATURE_SCHEMA_VERSION",
    "get_feature_schema",
    "encode_modulation",
    "encode_interleaver_family",
    "encode_fec_family",
    "encode_validation_status",
    "PipelineFeatureBuilder",
    "evaluate_candidate_correctness",
    "PipelineDatasetBuilder",
    "split_signals_by_group",
    "ScoreCalibrator",
    "extract_xgboost_feature_importance",
    "explain_candidate_prediction",
    "evaluate_signal_level_ranking",
    "evaluate_pipeline_model",
    "compute_expected_calibration_error",
    "train_pipeline_scorer",
    "rank_pipeline_candidates",
    "compute_ranking_uncertainty",
    "determine_confidence_tier",
    "PipelineScorer",
    "save_pipeline_model_package",
    "load_pipeline_model_package",
    "PipelineScoringEngine",
    "generate_synthetic_candidate_tree",
]
