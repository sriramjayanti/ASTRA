"""
ASTRA Symbol-Rate Estimation Demonstration Example.
Shows single signal estimation, explainability metrics, and batch estimation.
"""

import sys
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from astra_symbol_rate.src.inference import SymbolRateEstimator
from astra_symbol_rate.src.train import train_symbol_rate_ranker
from astra_symbol_rate.src.utils import generate_synthetic_signal


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("ASTRA Symbol-Rate (Baud) Estimation Engine - Production Demo")
    print("=" * 80)

    # 1. Ensure a trained ranker checkpoint exists
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "symbol_rate_ranker.joblib"
    if not ckpt_path.exists():
        print("\n[Step 1] No checkpoint found. Training baseline XGBoost ranker...")
        train_symbol_rate_ranker(num_signals=100, output_dir=str(ckpt_path.parent))

    # 2. Instantiate Estimator
    estimator = SymbolRateEstimator(model_path=str(ckpt_path))

    # 3. Simulate a QPSK Burst at 9600 Baud with RRC rolloff 0.25, SNR 18 dB, CFO 250 Hz
    print("\n[Step 2] Simulating incoming QPSK burst...")
    true_rate = 9600.0
    sample_rate = 192000.0
    iq, meta = generate_synthetic_signal(
        mod_type="QPSK",
        symbol_rate_hz=true_rate,
        sample_rate_hz=sample_rate,
        num_symbols=1024,
        snr_db=18.0,
        cfo_hz=250.0,
        rolloff=0.25,
    )
    print(f"True Modulation   : {meta['mod_type']}")
    print(f"True Symbol Rate  : {meta['true_symbol_rate_hz']} Baud")
    print(f"Sample Rate       : {meta['sample_rate_hz']} Hz")
    print(f"True SPS          : {meta['true_sps']:.2f}")
    print(f"Channel Impairment: SNR={meta['snr_db']} dB, CFO={meta['cfo_hz']} Hz")

    # 4. Optional Modulation Context from Fusion Engine
    modulation_context = {
        "top_k": [{"class": "QPSK", "probability": 0.86}, {"class": "8PSK", "probability": 0.09}],
        "probabilities": {"QPSK": 0.86, "8PSK": 0.09, "BPSK": 0.03, "16-QAM": 0.02},
        "confidence": 0.86,
    }

    # 5. Run Estimation
    print("\n[Step 3] Running SymbolRateEstimator...")
    prediction = estimator.estimate(
        iq=iq,
        sample_rate_hz=sample_rate,
        modulation_prediction=modulation_context,
    )

    # 6. Display Explainable Output
    print("\n" + "=" * 80)
    print("ASTRA SYMBOL-RATE PREDICTION RESULT")
    print("=" * 80)
    print(f"Best Symbol Rate  : {prediction.best_symbol_rate_hz:.2f} Baud")
    print(f"Samples Per Symbol: {prediction.samples_per_symbol:.2f} (Float SPS)")
    print(f"Confidence        : {prediction.confidence:.4f}")
    print(f"ASTRA Status      : {prediction.status}")
    print(f"Confidence Margin : {prediction.confidence_margin:.4f}")
    print(f"Model Version     : {prediction.model_version}")

    print("\nTop-K Candidate Hypotheses:")
    for cand in prediction.top_k:
        print(f"  Rank #{cand['rank']}: {cand['rate_hz']:8.1f} Baud (SPS={cand['samples_per_symbol']:5.2f}) | Score={cand['score']:.4f} | Supported by: {', '.join(cand['supported_by'])}")

    print("\nPreserved DSP Evidence:")
    ev = prediction.dsp_evidence
    print(f"  - Occupied Bandwidth : {ev.get('occupied_bandwidth_hz', 0):.1f} Hz")
    print(f"  - Estimated SNR      : {ev.get('estimated_snr_db', 0):.2f} dB")
    print(f"  - Estimated CFO      : {ev.get('cfo_estimate_hz', 0):.1f} Hz")
    print(f"  - Spectral Flatness  : {ev.get('spectral_flatness', 0):.4f}")

    print("\n" + "=" * 80)
    print("Demonstration successfully completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
