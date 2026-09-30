import tempfile
from pathlib import Path
import numpy as np

from astra_modulation_2d.src.splits import verify_disjoint_splits, save_split_json, load_split_json


def test_16_train_test_source_ids_disjoint():
    """TEST 16: Disjoint splits pass verification while overlapping splits raise ValueError."""
    train_ids = ["sig_001", "sig_002", "sig_003"]
    val_ids = ["sig_004", "sig_005"]
    test_ids = ["sig_006", "sig_007"]

    # Should pass without error
    verify_disjoint_splits(train_ids, val_ids, test_ids)

    # Contaminate train with test ID
    leaked_train = train_ids + ["sig_006"]
    failed = False
    try:
        verify_disjoint_splits(leaked_train, val_ids, test_ids)
    except ValueError:
        failed = True
    assert failed, "Expected ValueError on split leakage."


def test_17_same_split_as_supplied_manifest():
    """TEST 17: Split JSON saving and loading preserves identical ID membership."""
    train_ids = [f"sig_{i:04d}" for i in range(100)]
    val_ids = [f"sig_{i:04d}" for i in range(100, 120)]
    test_ids = [f"sig_{i:04d}" for i in range(120, 140)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        split_p = Path(tmp_dir) / "splits.json"
        save_split_json(split_p, train_ids, val_ids, test_ids)

        l_train, l_val, l_test = load_split_json(split_p)
        assert l_train == train_ids
        assert l_val == val_ids
        assert l_test == test_ids
