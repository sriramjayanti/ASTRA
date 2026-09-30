"""
ASTRA Symbol-Rate (Baud) Estimation Engine Package.
"""

from .candidate_features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_candidate_feature_matrix
from .candidate_generator import generate_candidates
from .dataset_builder import build_candidate_dataset_from_signals, extract_dsp_evidence
from .inference import SymbolRateEstimator
from .models import (
    DSPRateEvidence,
    InvalidSignalError,
    NoCandidatesFoundError,
    SymbolRateCandidate,
    SymbolRatePrediction,
)
from .preprocessing import preprocess_iq
from .xgboost_ranker import XGBoostSymbolRateRanker

__all__ = [
    "SymbolRateEstimator",
    "XGBoostSymbolRateRanker",
    "DSPRateEvidence",
    "SymbolRateCandidate",
    "SymbolRatePrediction",
    "InvalidSignalError",
    "NoCandidatesFoundError",
    "FEATURE_COLUMNS",
    "FEATURE_SCHEMA_VERSION",
    "preprocess_iq",
    "extract_dsp_evidence",
    "generate_candidates",
    "build_candidate_feature_matrix",
    "build_candidate_dataset_from_signals",
]
