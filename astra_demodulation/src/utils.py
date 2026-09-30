"""
utils.py
Synthetic test generators with bit ground truth for demodulation validation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from .mappings import get_constellation


def generate_synthetic_demod_test_data(
    modulation: str = "QPSK",
    num_symbols: int = 500,
    snr_db: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic symbol stream with exact bit and symbol ground truth.
    
    Returns:
        (rx_symbols, tx_symbols, tx_bits, metadata)
    """
    mod_clean = modulation.upper()

    if "FSK" in mod_clean:
        # FSK generation
        is_4fsk = "4" in mod_clean
        bps = 2 if is_4fsk else 1
        m_tones = 4 if is_4fsk else 2
        
        tx_symbols = np.random.randint(0, m_tones, num_symbols)
        if not is_4fsk:
            tx_bits = tx_symbols.astype(np.uint8)
            freq_devs = np.where(tx_symbols == 0, 1.0, -1.0)
        else:
            gray_map = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
            tx_bits = gray_map[tx_symbols].flatten()
            levels = np.array([-1.5, -0.5, 0.5, 1.5])
            freq_devs = levels[tx_symbols]

        # Convert frequency trajectory to complex phasor samples
        rx_symbols = np.exp(1j * np.cumsum(freq_devs * 0.5)).astype(np.complex64)
        tx_complex_symbols = rx_symbols.copy()
    else:
        # Linear digital modulations (PSK / QAM)
        const = get_constellation(modulation)
        bps = const.bits_per_symbol
        m_order = len(const.complex_points)

        symbol_indices = np.random.randint(0, m_order, num_symbols)
        tx_complex_symbols = const.complex_points[symbol_indices]
        tx_bits = const.bit_labels[symbol_indices].flatten().astype(np.uint8)
        rx_symbols = tx_complex_symbols.copy()

    # Add AWGN if specified
    if snr_db is not None:
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_pwr = 1.0 / snr_lin
        noise_std = np.sqrt(noise_pwr / 2.0)
        noise = (np.random.randn(len(rx_symbols)) + 1j * np.random.randn(len(rx_symbols))) * noise_std
        rx_symbols = (rx_symbols + noise).astype(np.complex64)

    meta = {
        "modulation": modulation,
        "num_symbols": num_symbols,
        "bits_per_symbol": bps,
        "total_bits": len(tx_bits),
        "snr_db": snr_db
    }
    return rx_symbols, tx_complex_symbols, tx_bits, meta
