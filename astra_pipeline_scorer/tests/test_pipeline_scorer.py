"""
test_pipeline_scorer.py
Master 30-Unit-Test Suite for ASTRA Stage 11 — Pipeline Scoring Model.
Covers all 30 tests mandated in specification Section 98.
"""

import pytest
import numpy as np
import tempfile
import json
import os

from astra_pipeline_scorer.src.models import (
    ConfidenceTier,
    PipelinePathCandidate,
    RankedCandidate,
    PipelineRankingResult,
    EvaluationReport,
)
from astra_pipeline_scorer.src.feature_schema import (
    PIPELINE_FEATURE_COLUMNS,
    FEATURE_DEFAULTS,
    PIPELINE_FEATURE_SCHEMA_VERSION,
    get_feature_schema,
)
from astra_pipeline_scorer.src.categorical import (
    encode_modulation,
    encode_interleaver_family,
    encode_fec_family,
    encode_validation_status,
)
from astra_pipeline_scorer.src.feature_builder import PipelineFeatureBuilder
from astra_pipeline_scorer.src.label_builder import evaluate_candidate_correctness
from astra_pipeline_scorer.src.dataset_builder import PipelineDatasetBuilder, split_signals_by_group
from astra_pipeline_scorer.src.calibration import ScoreCalibrator
from astra_pipeline_scorer.src.feature_importance import extract_xgboost_feature_importance, explain_candidate_prediction
from astra_pipeline_scorer.src.evaluation import evaluate_signal_level_ranking, evaluate_pipeline_model
from astra_pipeline_scorer.src.training import train_pipeline_scorer
from astra_pipeline_scorer.src.ranking import rank_pipeline_candidates, compute_ranking_uncertainty, determine_confidence_tier
from astra_pipeline_scorer.src.scorer import PipelineScorer
from astra_pipeline_scorer.src.checkpoint import save_pipeline_model_package, load_pipeline_model_package
from astra_pipeline_scorer.src.inference import PipelineScoringEngine
from astra_pipeline_scorer.src.utils import generate_synthetic_candidate_tree


# TEST 1: feature schema fixed
def test_1_feature_schema_fixed():
    schema = get_feature_schema()
    assert schema["schema_version"] == "pipeline_features_v1"
    assert len(PIPELINE_FEATURE_COLUMNS) == len(schema["columns"])


# TEST 2: feature extraction from candidate path
def test_2_feature_extraction_from_candidate_path():
    builder = PipelineFeatureBuilder()
    cands, _ = generate_synthetic_candidate_tree("sig_1", num_competing_candidates=2)
    vec = builder.extract_feature_vector(cands[0])
    assert len(vec) == len(PIPELINE_FEATURE_COLUMNS)


# TEST 3: missing feature handling
def test_3_missing_feature_handling():
    builder = PipelineFeatureBuilder()
    vec = builder.extract_feature_vector({})
    assert len(vec) == len(PIPELINE_FEATURE_COLUMNS)
    assert not np.isnan(vec).any()


# TEST 4: categorical encoding stable
def test_4_categorical_encoding_stable():
    assert encode_modulation("QPSK") == 1.0
    assert encode_interleaver_family("block") == 1.0
    assert encode_fec_family("convolutional") == 1.0
    assert encode_validation_status("VALIDATION_STRONG") == 4.0


# TEST 5: training dataset generation
def test_5_training_dataset_generation():
    cands, _ = generate_synthetic_candidate_tree("sig_test", num_competing_candidates=6)
    builder = PipelineDatasetBuilder()
    dataset = builder.build_dataset(cands, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    assert "X_train" in dataset and "y_train" in dataset


# TEST 6: candidate labels correct
def test_6_candidate_labels_correct():
    gt = {"modulation": "QPSK", "symbol_rate_hz": 9600.0, "interleaver_family": "block", "fec_family": "convolutional"}
    right = {"modulation": "QPSK", "symbol_rate_hz": 9600.0, "interleaver_family": "block", "fec_family": "convolutional"}
    wrong = {"modulation": "16QAM", "symbol_rate_hz": 9600.0, "interleaver_family": "block", "fec_family": "convolutional"}
    assert evaluate_candidate_correctness(right, gt) == 1
    assert evaluate_candidate_correctness(wrong, gt) == 0


# TEST 7: truth excluded from feature X
def test_7_truth_excluded_from_feature_x():
    forbidden = ["true_modulation", "true_symbol_rate", "candidate_correct", "ground_truth"]
    for col in PIPELINE_FEATURE_COLUMNS:
        assert col not in forbidden


# TEST 8: source group split has no leakage
def test_8_source_group_split_has_no_leakage():
    sigs = [f"sig_{i}" for i in range(30)]
    tr, va, te = split_signals_by_group(sigs, 0.6, 0.2, 0.2)
    assert len(set(tr).intersection(set(va))) == 0
    assert len(set(tr).intersection(set(te))) == 0


# TEST 9: XGBoost training runs
def test_9_xgboost_training_runs():
    all_cands = []
    for i in range(10):
        c, _ = generate_synthetic_candidate_tree(f"s_{i}", num_competing_candidates=6, seed=i)
        all_cands.extend(c)
    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, 0.7, 0.3, 0.0)
    model, calib, rep, meta = train_pipeline_scorer(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    assert model is not None


# TEST 10: checkpoint save/load
def test_10_checkpoint_save_and_load():
    all_cands = []
    for i in range(8):
        c, _ = generate_synthetic_candidate_tree(f"s_{i}", num_competing_candidates=6, seed=i)
        all_cands.extend(c)
    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, 0.7, 0.3, 0.0)
    model, calib, _, _ = train_pipeline_scorer(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_pipeline_model_package(model, calib, tmp_dir)
        m, c, meta = load_pipeline_model_package(tmp_dir)
        assert m is not None


# TEST 11: inference feature order preserved
def test_11_inference_feature_order_preserved():
    builder = PipelineFeatureBuilder()
    vec = builder.extract_feature_vector({"modulation": "8PSK"})
    assert len(vec) == len(PIPELINE_FEATURE_COLUMNS)
    assert vec[0] == encode_modulation("8PSK") # 'mod_encoded' is column 0


# TEST 12: candidate scores finite
def test_12_candidate_scores_finite():
    scorer = PipelineScorer()
    cands, _ = generate_synthetic_candidate_tree("s1", num_competing_candidates=4)
    scores = scorer.score_batch(cands)
    assert np.all(np.isfinite(scores))


# TEST 13: candidate ranking descending
def test_13_candidate_ranking_descending():
    cands = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]
    scores = np.array([0.2, 0.8])
    res = rank_pipeline_candidates(cands, scores)
    assert res.ranked_candidates[0].candidate_id == "c2"
    assert res.ranked_candidates[1].candidate_id == "c1"


# TEST 14: Top-K correct length
def test_14_top_k_correct_length():
    cands = [{"candidate_id": f"c_{i}"} for i in range(10)]
    scores = np.linspace(0.1, 0.9, 10)
    res = rank_pipeline_candidates(cands, scores, top_k=3)
    assert len(res.ranked_candidates) == 3


# TEST 15: score margin calculated
def test_15_score_margin_calculated():
    cands = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]
    scores = np.array([0.9, 0.6])
    res = rank_pipeline_candidates(cands, scores)
    assert abs(res.score_margin - 0.3) < 1e-4


# TEST 16: batch scoring
def test_16_batch_scoring():
    scorer = PipelineScorer()
    cands, _ = generate_synthetic_candidate_tree("s1", num_competing_candidates=10)
    scores = scorer.score_batch(cands)
    assert len(scores) == 10


# TEST 17: candidate grouping by signal
def test_17_candidate_grouping_by_signal():
    sids = ["sigA", "sigA", "sigB", "sigB"]
    scores = np.array([0.9, 0.1, 0.3, 0.7])
    labels = np.array([1, 0, 0, 1])
    res = evaluate_signal_level_ranking(sids, scores, labels)
    assert res["total_signals"] == 2
    assert res["top1_accuracy"] == 1.0


# TEST 18: candidate recall metric
def test_18_candidate_recall_metric():
    sids = ["sigA", "sigB"]
    scores = np.array([0.9, 0.8])
    labels = np.array([1, 0]) # sigB has no truth
    res = evaluate_signal_level_ranking(sids, scores, labels)
    assert res["candidate_recall"] == 0.5


# TEST 19: conditional ranking accuracy
def test_19_conditional_ranking_accuracy():
    sids = ["sigA", "sigB"]
    scores = np.array([0.9, 0.8])
    labels = np.array([1, 0])
    res = evaluate_signal_level_ranking(sids, scores, labels)
    assert res["conditional_top1_accuracy"] == 1.0


# TEST 20: Top-1 accuracy
def test_20_top_1_accuracy():
    sids = ["sigA"]
    scores = np.array([0.9, 0.2])
    labels = np.array([1, 0])
    res = evaluate_signal_level_ranking(sids, scores, labels)
    assert res["top1_accuracy"] == 1.0


# TEST 21: Top-3 success
def test_21_top_3_success():
    sids = ["sigA", "sigA", "sigA", "sigA"]
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    labels = np.array([0, 0, 1, 0]) # 3rd position
    res = evaluate_signal_level_ranking(sids, scores, labels)
    assert res["top3_accuracy"] == 1.0
    assert res["top1_accuracy"] == 0.0


# TEST 22: MRR calculation
def test_22_mrr_calculation():
    sids = ["sigA", "sigA"]
    scores = np.array([0.9, 0.8])
    labels = np.array([0, 1]) # rank 2 -> MRR = 0.5
    res = evaluate_signal_level_ranking(sids, scores, labels)
    assert res["mrr"] == 0.5


# TEST 23: class imbalance handling
def test_23_class_imbalance_handling():
    all_cands = []
    for i in range(10):
        c, _ = generate_synthetic_candidate_tree(f"s_{i}", num_competing_candidates=10, seed=i)
        all_cands.extend(c) # 1 positive per 10
    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, 0.7, 0.3, 0.0)
    model, _, _, meta = train_pipeline_scorer(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    assert meta["scale_pos_weight"] > 1.0


# TEST 24: hard negatives retained
def test_24_hard_negatives_retained():
    cands, _ = generate_synthetic_candidate_tree("s1", num_competing_candidates=8)
    cand_types = [c.candidate_id for c in cands]
    assert "cand_wrong_inter" in cand_types
    assert "cand_wrong_fec" in cand_types


# TEST 25: calibration pipeline
def test_25_calibration_pipeline():
    calib = ScoreCalibrator(method="isotonic")
    calib.fit(np.array([0.1, 0.9]), np.array([0, 1]))
    res = calib.calibrate(np.array([0.8]))
    assert len(res) == 1


# TEST 26: feature importance generated
def test_26_feature_importance_generated():
    all_cands = []
    for i in range(8):
        c, _ = generate_synthetic_candidate_tree(f"s_{i}", num_competing_candidates=6, seed=i)
        all_cands.extend(c)
    builder = PipelineDatasetBuilder()
    data = builder.build_dataset(all_cands, 0.7, 0.3, 0.0)
    model, _, _, _ = train_pipeline_scorer(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    imp = extract_xgboost_feature_importance(model)
    assert len(imp) > 0


# TEST 27: rule-based fallback
def test_27_rule_based_fallback():
    engine = PipelineScoringEngine()
    cands, _ = generate_synthetic_candidate_tree("s1", num_competing_candidates=4)
    res = engine.rank(cands, signal_id="s1")
    assert res.fallback_used is True
    assert res.ranked_candidates[0].candidate_id == "cand_correct"


# TEST 28: model version emitted
def test_28_model_version_emitted():
    engine = PipelineScoringEngine()
    cands, _ = generate_synthetic_candidate_tree("s1", num_competing_candidates=3)
    res = engine.rank(cands)
    assert res.model_version == "astra_pipeline_xgb_v1"


# TEST 29: schema mismatch rejected
def test_29_schema_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmp_dir:
        schema_path = os.path.join(tmp_dir, "feature_schema.json")
        with open(schema_path, "w") as f:
            json.dump({"schema_version": "wrong_v99"}, f)
        with pytest.raises(ValueError):
            load_pipeline_model_package(tmp_dir)


# TEST 30: JSON result serialization
def test_30_json_result_serialization():
    engine = PipelineScoringEngine()
    cands, _ = generate_synthetic_candidate_tree("s1", num_competing_candidates=4)
    res = engine.rank(cands)
    d = res.to_dict()
    json_str = json.dumps(d)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert parsed["top1_candidate_id"] == "cand_correct"
