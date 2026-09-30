import numpy as np
import json
import sys

manifest = "datasets/ASTRA_FINAL_TEST_SET_V2_BLIND/blind_benchmark_manifest.json"
with open(manifest) as f:
    d = json.load(f)

for cap in d["captures"]:
    if cap["true_modulation"] in ["BPSK", "QPSK", "16QAM"]:
        iq = np.fromfile(cap["iq_path"], dtype=np.complex64)
        sr = float(cap["sample_rate"])
        true_cfo = float(cap["cfo_hz"])
        mod = cap["true_modulation"]
        m = 2 if mod == "BPSK" else 4
        
        fft_size = 4096
        # Without zero-mean DC removal on raised:
        raised = (iq[:fft_size] ** m) * np.blackman(fft_size)
        fft_out = np.fft.fftshift(np.fft.fft(raised, n=fft_size))
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sr))
        mag = np.abs(fft_out)
        center_idx = fft_size // 2
        
        print(f"\n{mod} (True CFO = {true_cfo:.1f} Hz, m*CFO = {m*true_cfo:.1f} Hz, Fs = {sr:.1f}):")
        print(f"  Center (0 Hz) magnitude: {mag[center_idx]:.2f}")
        print(f"  Max magnitude: {np.max(mag):.2f} at frequency {freqs[np.argmax(mag)]:.1f} Hz")
        
        # What if we remove mean of raised or zero out DC bin?
        raised_centered = (iq[:fft_size] ** m)
        raised_centered = (raised_centered - np.mean(raised_centered)) * np.blackman(fft_size)
        fft_c = np.fft.fftshift(np.fft.fft(raised_centered, n=fft_size))
        mag_c = np.abs(fft_c)
        peak_c_freq = freqs[np.argmax(mag_c)]
        print(f"  After DC removal on raised: Peak freq = {peak_c_freq:.1f} Hz -> Est CFO = {peak_c_freq/m:.1f} Hz")
