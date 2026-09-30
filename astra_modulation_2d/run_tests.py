"""
Test runner for ASTRA 2D Spectrogram Classifier unit tests.
"""

import sys
import traceback
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from astra_modulation_2d.tests.test_spectrogram import (
    test_1_stft_output_finite,
    test_2_spectrogram_deterministic,
    test_3_spectrogram_dimensions_correct,
    test_4_fftshift_correct,
    test_5_dc_removal_correct,
    test_6_rms_normalization_correct,
    test_19_nan_inf_handling,
)
from astra_modulation_2d.tests.test_model import (
    test_7_model_input_shape_accepted,
    test_8_model_output_logits,
    test_9_extract_features_shape,
    test_20_variable_batch_size,
)
from astra_modulation_2d.tests.test_inference import (
    test_10_probabilities_sum_to_one,
    test_11_top_k_sorted,
    test_12_cpu_inference,
    test_13_cuda_inference_if_available,
    test_18_batch_inference,
)
from astra_modulation_2d.tests.test_checkpoint import (
    test_14_checkpoint_save_and_load,
    test_15_class_mapping_preserved_and_mismatch_fails,
)
from astra_modulation_2d.tests.test_split_leakage import (
    test_16_train_test_source_ids_disjoint,
    test_17_same_split_as_supplied_manifest,
)
from astra_modulation_2d.tests.test_dataset import test_dataset_item_generation

test_funcs = [
    ("TEST 1: STFT output finite", test_1_stft_output_finite),
    ("TEST 2: Spectrogram deterministic", test_2_spectrogram_deterministic),
    ("TEST 3: Spectrogram dimensions correct", test_3_spectrogram_dimensions_correct),
    ("TEST 4: fftshift correct", test_4_fftshift_correct),
    ("TEST 5: DC removal correct", test_5_dc_removal_correct),
    ("TEST 6: RMS normalization correct", test_6_rms_normalization_correct),
    ("TEST 7: Model input shape accepted", test_7_model_input_shape_accepted),
    ("TEST 8: Model output [B, num_classes]", test_8_model_output_logits),
    ("TEST 9: extract_features [B, 256]", test_9_extract_features_shape),
    ("TEST 10: Probabilities sum approximately 1", test_10_probabilities_sum_to_one),
    ("TEST 11: Top-K sorted", test_11_top_k_sorted),
    ("TEST 12: CPU inference", test_12_cpu_inference),
    ("TEST 13: CUDA inference", test_13_cuda_inference_if_available),
    ("TEST 14: Checkpoint save and load", test_14_checkpoint_save_and_load),
    ("TEST 15: Class mapping preserved", test_15_class_mapping_preserved_and_mismatch_fails),
    ("TEST 16: Train/test source IDs disjoint", test_16_train_test_source_ids_disjoint),
    ("TEST 17: Same split as supplied split manifest", test_17_same_split_as_supplied_manifest),
    ("TEST 18: Batch inference", test_18_batch_inference),
    ("TEST 19: NaN/Inf handling", test_19_nan_inf_handling),
    ("TEST 20: Variable batch size", test_20_variable_batch_size),
    ("TEST 21: Dataset item generation", test_dataset_item_generation),
]

passed = 0
failed = 0

print("=" * 80)
print(f"RUNNING ASTRA 2D SPECTROGRAM CLASSIFIER TEST SUITE ({len(test_funcs)} TESTS)")
print("=" * 80)

for name, func in test_funcs:
    try:
        func()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

print("=" * 80)
print(f"RESULTS: {passed} PASSED, {failed} FAILED across {len(test_funcs)} tests.")
print("=" * 80)

if failed > 0:
    sys.exit(1)
