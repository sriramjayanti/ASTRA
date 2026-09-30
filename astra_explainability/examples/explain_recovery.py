"""
ASTRA STAGE 15: Explainability & Confidence Reasoning Engine
Example Walkthrough: End-to-End Explanation for Recovered Signal ("hi hello")
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from astra_explainability.src.inference import ExplainabilityEngine
from astra_explainability.src.utils import (
    create_synthetic_perfect_record,
    create_low_snr_record,
    create_wrong_top1_model_record,
    create_user_override_record
)


def run_explainability_demo():
    print("=" * 80)
    print("ASTRA STAGE 15: EXPLAINABILITY & CONFIDENCE REASONING ENGINE")
    print("=" * 80)

    engine = ExplainabilityEngine()

    # 1. Master Demonstration: Synthetic Recovered Signal ("hi hello")
    print("\n[SCENARIO 1] Full Downstream Validated Recovery ('hi hello')")
    print("-" * 80)
    record = create_synthetic_perfect_record()
    res = engine.explain(record)

    print(f"Signal ID         : {res.signal_id}")
    print(f"Top Pipeline ID   : {res.best_pipeline_id}")
    print(f"Overall Status    : {res.overall_status.value}")
    print(f"ASTRA Confidence  : {res.overall_confidence:.4f}")

    print("\n--- Canonical Field Explanations ---")
    header = f"{'Field':<18} | {'Value':<26} | {'Status':<10} | {'Confidence':<10} | {'Sources':<20}"
    print(header)
    print("-" * len(header))
    for fname, fexp in res.field_explanations.items():
        v_str = str(fexp.value) if fexp.value is not None else "None"
        if len(v_str) > 24:
            v_str = v_str[:21] + "..."
        src_str = ", ".join(fexp.source_stages)
        if len(src_str) > 20:
            src_str = src_str[:17] + "..."
        print(f"{fname:<18} | {v_str:<26} | {fexp.status.value:<10} | {fexp.confidence_score:<10.2f} | {src_str:<20}")

    print("\n--- Human-Readable Narrative Summary ---")
    print(res.human_summary)

    print("\n--- Candidate Pipeline Comparison (Top-3) ---")
    for cand in res.candidate_comparison:
        best_mark = " [WINNER]" if cand.is_best else ""
        print(f"Rank #{cand.rank} [{cand.pipeline_id}]{best_mark} - Score: {cand.score:.4f} (Margin: {cand.margin_to_next:.4f})")
        if cand.why_lost:
            print(f"  Reason: {cand.why_lost}")

    print("\n--- GUI Summary Cards ---")
    for card in res.gui_summary.cards:
        print(f"  [{card.badge_color.upper():<6}] {card.title:<16}: {str(card.primary_value):<20} (Conf: {card.confidence_pct:.1f}%)")

    # 2. Scenario 2: Early ML Disagreement Overridden by Downstream CRC
    print("\n" + "=" * 80)
    print("[SCENARIO 2] Early Model Preferred 8PSK, But Downstream CRC Proved QPSK")
    print("-" * 80)
    record_wrong = create_wrong_top1_model_record()
    res_wrong = engine.explain(record_wrong)
    mod_exp = res_wrong.field_explanations["modulation"]
    print(f"Selected Modulation: {mod_exp.value} (Status: {mod_exp.status.value}, Conf: {mod_exp.confidence_score:.2f})")
    print("Supporting Evidence:")
    for s in mod_exp.supporting_evidence:
        print(f"  + {s}")
    print("Contradicting Evidence (captured from early disagreement):")
    for c in mod_exp.contradicting_evidence:
        print(f"  - {c}")

    # 3. Scenario 3: Expert User Override Conflict
    print("\n" + "=" * 80)
    print("[SCENARIO 3] Expert User Override Conflict (User forced 16QAM over QPSK)")
    print("-" * 80)
    record_override = create_user_override_record()
    res_override = engine.explain(record_override)
    print(f"Contradictions Detected: {len(res_override.contradictions)}")
    for c in res_override.contradictions:
        print(f"  [{c.severity.value}] {c.contradiction_id}: {c.description}")
        print(f"     Evidence A: {c.evidence_a}")
        print(f"     Evidence B: {c.evidence_b}")

    print("\n" + "=" * 80)
    print("EXPLAINABILITY ENGINE DEMONSTRATION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    run_explainability_demo()
