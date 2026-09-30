"""
ASTRA Random Forest Support Model Demonstration Example.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from astra_random_forest.src.inference import RandomForestSupportEngine
from astra_random_forest.src.train import train_random_forest_support
from astra_random_forest.src.utils import generate_impaired_signal


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("ASTRA Random Forest Support Model - Demonstration")
    print("=" * 80)

    # 1. Ensure trained checkpoint exists
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "random_forest_support.joblib"
    if not ckpt_path.exists():
        print("\n[Step 1] Training baseline Random Forest models...")
        train_random_forest_support(num_signals=120, output_dir=str(ckpt_path.parent))

    # 2. Instantiate Engine
    engine = RandomForestSupportEngine(checkpoint_path=str(ckpt_path))

    # 3. Simulate Impaired 16-QAM signal (Low SNR = 2 dB, CFO = 750 Hz)
    print("\n[Step 2] Simulating incoming 16-QAM burst with channel impairments...")
    iq_qam, meta = generate_impaired_signal(
        mod_type="16-QAM",
        sample_rate_hz=192000.0,
        symbol_rate_hz=9600.0,
        num_symbols=1024,
        snr_db=2.0,
        cfo_hz=750.0,
        clipping_ratio=0.08,
        multipath=False,
        seed=101,
    )
    print(f"  True Modulation : {meta['mod_type']}")
    print(f"  True SNR        : {meta['snr_db']} dB")
    print(f"  True CFO        : {meta['cfo_hz']} Hz")
    print(f"  Clipping Ratio  : {meta['clipping_ratio']}")

    # 4. Predict Broad Family and Quality Evidence
    print("\n[Step 3] Running Random Forest Inference...")
    result = engine.predict_all(iq_qam, sample_rate_hz=192000.0)

    print("\n" + "=" * 80)
    print("RANDOM FOREST PREDICTION RESULTS")
    print("=" * 80)
    print(f"Predicted Family  : {result['family']}")
    print(f"Confidence        : {result['confidence']:.4f}")
    print(f"Status            : {result['status']}")
    print(f"Confidence Margin : {result['confidence_margin']:.4f}")

    print("\nFamily Probability Distribution:")
    for fam_name, prob in result["probabilities"].items():
        print(f"  - {fam_name:8s}: {prob:.4f}")

    print("\nSignal Quality & Impairment Evidence:")
    for qual_name, prob in result["quality"].items():
        flag = "DETECTED" if prob >= 0.50 else "nominal"
        print(f"  - {qual_name:25s}: {prob:.4f} ({flag})")

    print("\nTop Contributing DSP Features:")
    for item in result["important_evidence"]:
        print(f"  - {item['feature']:28s}: {item['importance']:.4f}")

    print("\n" + "=" * 80)
    print("Demonstration successfully completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
