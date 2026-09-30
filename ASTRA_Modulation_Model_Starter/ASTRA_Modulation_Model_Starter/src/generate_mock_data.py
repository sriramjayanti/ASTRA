import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

from dataset import DEFAULT_CLASSES


def generate_qam_constellation(order: int) -> np.ndarray:
    """Generates standard square QAM constellation symbols."""
    m_side = int(np.sqrt(order))
    coords = np.arange(-(m_side - 1), m_side, 2)
    constellation = []
    for real in coords:
        for imag in coords:
            constellation.append(real + 1j * imag)
    constellation = np.array(constellation, dtype=np.complex64)
    # Normalize average power to 1.0
    return constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))


def generate_psk_constellation(order: int) -> np.ndarray:
    """Generates PSK constellation symbols on unit circle."""
    phases = 2 * np.pi * np.arange(order) / order
    return np.exp(1j * phases).astype(np.complex64)


def generate_synthetic_signal(
    modulation: str,
    num_samples: int = 16384,
    snr_db: float = 10.0,
    cfo: float = 0.01,
    sps: int = 4
) -> np.ndarray:
    """
    Generates a realistic synthetic complex baseband signal with:
    - Target modulation constellation
    - Upsampling and pulse shaping filter
    - Carrier Frequency Offset (CFO)
    - AWGN noise at target SNR
    """
    num_symbols = num_samples // sps + 100

    mod = modulation.lower().strip()
    if mod == "bpsk":
        symbols = np.random.choice(generate_psk_constellation(2), size=num_symbols)
    elif mod == "qpsk":
        symbols = np.random.choice(generate_psk_constellation(4), size=num_symbols)
    elif mod == "8psk":
        symbols = np.random.choice(generate_psk_constellation(8), size=num_symbols)
    elif mod == "dqpsk":
        base_syms = np.random.choice(generate_psk_constellation(4), size=num_symbols)
        # Differential encoding: s[k] = s[k-1] * sym[k]
        symbols = np.zeros(num_symbols, dtype=np.complex64)
        current = 1.0 + 0j
        for i in range(num_symbols):
            current = current * base_syms[i]
            symbols[i] = current
    elif mod == "msk":
        bits = np.random.randint(0, 2, size=num_symbols * 2) * 2 - 1
        # Simplified continuous-phase FSK / MSK
        phase = np.cumsum(bits * (np.pi / (2 * sps)))
        signal_base = np.exp(1j * phase[:num_samples]).astype(np.complex64)
    elif mod == "16qam":
        symbols = np.random.choice(generate_qam_constellation(16), size=num_symbols)
    elif mod == "64qam":
        symbols = np.random.choice(generate_qam_constellation(64), size=num_symbols)
    elif mod == "256qam":
        symbols = np.random.choice(generate_qam_constellation(256), size=num_symbols)
    else:
        symbols = np.random.choice(generate_psk_constellation(4), size=num_symbols)

    if mod != "msk":
        # Upsample
        upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
        upsampled[::sps] = symbols

        # Pulse shaping (Raised Cosine / RRC approximation via FIR)
        num_taps = 8 * sps + 1
        h_rrc = signal.firwin(num_taps, 1.0 / sps, window="hamming")
        shaped = signal.convolve(upsampled, h_rrc, mode="same")
        signal_base = shaped[:num_samples]

    # Carrier Frequency Offset
    t = np.arange(len(signal_base))
    signal_cfo = signal_base * np.exp(1j * 2 * np.pi * cfo * t)

    # Additive White Gaussian Noise (AWGN)
    signal_power = np.mean(np.abs(signal_cfo) ** 2) + 1e-12
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2.0) * (
        np.random.randn(len(signal_cfo)) + 1j * np.random.randn(len(signal_cfo))
    )

    rx_signal = signal_cfo + noise
    return rx_signal.astype(np.complex64)


def save_tim_file(filepath: Path, complex_signal: np.ndarray, dtype: str = "float32"):
    """
    Saves complex IQ signal as interleaved binary (.tim format).
    I0, Q0, I1, Q1, ...
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float32 if dtype == "float32" else np.float64
    
    i_part = np.real(complex_signal).astype(np_dtype)
    q_part = np.imag(complex_signal).astype(np_dtype)
    
    interleaved = np.empty((2 * len(complex_signal),), dtype=np_dtype)
    interleaved[0::2] = i_part
    interleaved[1::2] = q_part
    
    interleaved.tofile(str(filepath))


def generate_mock_dataset(
    output_dir: str = "data",
    num_signals_per_class: int = 15,
    samples_per_signal: int = 16384,
    tim_dtype: str = "float32"
):
    """
    Generates a full synthetic CSPB dataset for testing and milestone verification.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    sig_idx = 0

    snr_range = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]

    for mod in DEFAULT_CLASSES:
        for k in range(num_signals_per_class):
            snr_val = float(np.random.choice(snr_range))
            cfo_val = float(np.random.uniform(-0.02, 0.02))
            rolloff = float(np.random.choice([0.2, 0.25, 0.35]))
            t0_val = float(np.random.choice([2.0, 4.0, 8.0]))
            
            sig = generate_synthetic_signal(
                modulation=mod,
                num_samples=samples_per_signal,
                snr_db=snr_val,
                cfo=cfo_val,
                sps=4
            )

            filename = f"signal_{sig_idx:04d}.tim"
            file_path = out_dir / filename
            save_tim_file(file_path, sig, dtype=tim_dtype)

            records.append({
                "signal_index": f"signal_{sig_idx:04d}",
                "modulation": mod,
                "t0": t0_val,
                "carrier_offset": cfo_val,
                "rolloff": rolloff,
                "u": 1,
                "d": 1,
                "snr_db": snr_val,
                "noise_density_db": -174.0 + snr_val
            })
            sig_idx += 1

    truth_df = pd.DataFrame(records)
    truth_csv_path = out_dir / "truth.csv"
    truth_df.to_csv(truth_csv_path, index=False)
    print(f"Generated {sig_idx} synthetic signal files in {out_dir}")
    print(f"Truth metadata saved to {truth_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mock CSPB Dataset")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--signals_per_class", type=int, default=15)
    parser.add_argument("--samples_per_signal", type=int, default=16384)
    args = parser.parse_args()

    generate_mock_dataset(
        output_dir=args.output_dir,
        num_signals_per_class=args.signals_per_class,
        samples_per_signal=args.samples_per_signal
    )
