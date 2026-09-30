# CSPB.ML.2018R2 Truth Schema

1. signal_index — maps to signal_N.tim
2. modulation — target label
3. T0 — base symbol period
4. carrier_offset — normalized CFO
5. rolloff — RRC excess bandwidth
6. U — upsample factor
7. D — downsample factor
8. snr_db — in-band SNR
9. noise_density_db — noise spectral density

Normalized symbol rate:
f_sym = (1/T0) * (D/U)

Use modulation as Model 1 target.
Retain SNR for accuracy-vs-SNR evaluation.
Do not invent unsupported labels such as FEC, interleaver, header or payload.
