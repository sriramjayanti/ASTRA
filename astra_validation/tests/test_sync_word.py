"""
test_sync_word.py
Unit tests for sync word detection and periodicity scoring.
"""

import pytest
import numpy as np
from astra_validation.src.sync_word import search_sync_word, score_sync_periodicity, hex_to_bits
from astra_validation.src.models import EvidenceCheckState


def test_exact_sync_word_detection():
    # Embed 0xEB90 at 0, 256, 512, 768
    sync_bits = hex_to_bits("EB90")
    total_len = 1024
    stream = np.zeros(total_len, dtype=np.uint8)

    for pos in [0, 256, 512, 768]:
        stream[pos:pos + len(sync_bits)] = sync_bits

    res = search_sync_word(stream, "EB90", pattern_id="sync_EB90")
    assert res.match_count == 4
    assert res.positions == [0, 256, 512, 768]
    assert res.mean_hamming_distance == 0.0
    assert res.periodicity_score >= 0.90
    assert res.check_state == EvidenceCheckState.PASS


def test_hamming_tolerant_sync_detection():
    # Embed 0xEB90 with 1 flipped bit
    sync_bits = hex_to_bits("EB90").copy()
    sync_bits[3] ^= 1 # Corrupt 1 bit

    stream = np.zeros(512, dtype=np.uint8)
    stream[0:len(sync_bits)] = sync_bits
    stream[256:256 + len(sync_bits)] = sync_bits

    res = search_sync_word(stream, "EB90", pattern_id="sync_EB90", max_hamming_fraction=0.125)
    assert res.match_count == 2
    assert res.mean_hamming_distance == 1.0
    assert res.check_state == EvidenceCheckState.PASS
