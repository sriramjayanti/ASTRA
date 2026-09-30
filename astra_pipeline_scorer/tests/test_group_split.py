"""
test_group_split.py
Unit tests for group-aware splitting and leakage prevention.
"""

import pytest
from astra_pipeline_scorer.src.dataset_builder import split_signals_by_group


def test_source_group_split_has_no_leakage():
    signal_ids = [f"sig_{i}" for i in range(20) for _ in range(10)]
    train_sigs, val_sigs, test_sigs = split_signals_by_group(signal_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    s_train = set(train_sigs)
    s_val = set(val_sigs)
    s_test = set(test_sigs)

    assert len(s_train.intersection(s_val)) == 0
    assert len(s_train.intersection(s_test)) == 0
    assert len(s_val.intersection(s_test)) == 0
    assert len(s_train) + len(s_val) + len(s_test) == 20
