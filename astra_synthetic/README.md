# ASTRA — Synthetic Signal Generation Engines

**Automated Signal Analysis & Recovery Assistant (ASTRA)**  
*Production-quality synthetic dataset generation pipeline for blind signal recovery, detection, and receiver validation.*

---

## 1. Master Pipeline Architecture

```
┌────────────────────────────────────────────────────────┐
│  Engine 1: Payload / Bitstream Generator               │
│  - Ground-truth source bits (Random, Text, Counter...) │
└──────────────────────────┬─────────────────────────────┘
                           │ PayloadRecord.payload_bits
                           ▼
┌────────────────────────────────────────────────────────┐
│  Engine 2: Frame / Sync / Header / CRC Generator       │
│  - SYNC | HEADER | PAYLOAD | CRC                       │
│  - Ground-truth region boundaries & stream tracking    │
└──────────────────────────┬─────────────────────────────┘
                           │ FrameRecord.frame_bits
                           ▼
┌────────────────────────────────────────────────────────┐
│  Engine 3: FEC Encoding Generator                      │
│  - None, Convolutional, Reed-Solomon, Concatenated, LDPC│
│  - Ground-truth code parameters, rates & parity telemetry│
└──────────────────────────┬─────────────────────────────┘
                           │ FECRecord.encoded_bits
                           ▼
┌────────────────────────────────────────────────────────┐
│  Engine 4: Interleaver Generator                       │
│  - None, Block, Convolutional, Diagonal, Pseudo-Random │
│  - Exact permutations, delay lines, and flush telemetry │
└──────────────────────────┬─────────────────────────────┘
                           │ InterleaverRecord.interleaved_bits
                           ▼
┌────────────────────────────────────────────────────────┐
│  Engine 5: Modulation / Clean IQ Waveform Generator    │
│  - 2-FSK, 4-FSK, MSK, BPSK, QPSK, 8PSK, 16QAM, 64QAM,  │
│    256QAM                                              │
│  - Gray mapping, true RRC pulse shaping, complex64 IQ  │
└──────────────────────────┬─────────────────────────────┘
                           │ ModulationRecord.clean_iq
                           ▼
┌────────────────────────────────────────────────────────┐
│  Engine 6: RF / Channel Impairment Generator           │
│  - AWGN, CFO, Phase, Timing Offset, Gain, Drift,       │
│  - Rayleigh & Rician Fading, Multipath, Interference,   │
│  - Sample Clock Offset, Power Stages, Hashes           │
└──────────────────────────┬─────────────────────────────┘
                           │ ChannelRecord.impaired_iq
                           ▼
┌────────────────────────────────────────────────────────┐
│  Engine 7: IQ Capture & File Writer Generator          │
│  - Raw .iq/.bin (float32, int16, int8, LE/BE, IQ/QI)   │
│  - 2-channel stereo WAV (int16, float32, IQ/QI)        │
│  - SigMF dataset (.sigmf-data + .sigmf-meta)           │
│  - Quantization distortion telemetry & clipping counters│
└──────────────────────────┬─────────────────────────────┘
                           │ CaptureRecord
                           ▼
┌────────────────────────────────────────────────────────┐
│  Engine 8: Ground-Truth / Dataset Orchestrator         │
│  - Master Seed Hierarchy (SeedSequence)                │
│  - Source Chain ID tracking across all 7 stages        │
│  - Zero-leakage source-level Train/Val/Test splitting  │
│  - Balanced class generation & anti-shortcut sampling  │
│  - Separate visible metadata vs complete hidden truth  │
│  - Task manifests (Modulation, FEC, Interleaver, Bits) │
│  - Blind test package & Secret truth scoring package   │
│  - Automated clean-control end-to-end demod validation │
│  - Quality, leakage & distribution JSON reporting     │
└────────────────────────────────────────────────────────┘
```

---

## 2. Engine Breakdown

### Synthetic Engine 1 — Payload / Bitstream Generator
Generates controlled ground-truth source bits across diverse data characteristics:
- **Random Binary (`random_bits`)**: Uniform Bernoulli(0.5) bits.
- **Random Bytes (`random_bytes`)**: Uniform byte values mapped MSB-first.
- **ASCII / Text (`text`)**: UTF-8 encoded text strings.
- **Repeated Pattern (`repeated_pattern`)**: Periodic bit sequences (e.g., `10110011`).
- **Counter / Sequential (`counter`)**: Increasing byte counter `(start + i) % 256`.
- **Biased / Low-Entropy (`biased_random`)**: Controllable $P(1) = p$.
- **High-Entropy (`high_entropy`)**: Statistically balanced random data.
- **Fixed (`fixed_hex`, `fixed_bits`)**: Deterministic test patterns.

---

### Synthetic Engine 2 — Frame / Sync / Header / CRC Generator
Encapsulates raw payload bits into complete communication frames with exact boundary tracking:
- **Layout**: `| SYNC | HEADER | PAYLOAD | CRC |`
- **Sync Words**: Fixed hex/binary words, pool selection, seed-reproducible random sync words.
- **Structured Header**: MSB-first packing of `version`, `frame_type`, `sequence_number`, `payload_length`, and `flags`.
- **Documented CRC**: CRC-8, CRC-16-CCITT (`0x29B1`), CRC-32 (`0xCBF43926`).
- **Continuous Streams**: Concatenated multi-frame streams with optional inter-frame gaps.

---

### Synthetic Engine 3 — FEC Encoding / Coding Ground-Truth Generator
Applies channel coding to `FrameRecord.frame_bits` and outputs a `FECRecord`:
- **None (`none`)**: Direct uncoded pass-through.
- **Convolutional (`conv_k3_r12`, `conv_k5_r12`, `conv_k7_r12`)**: Shift-register encoding with zero-tail termination ($K-1$ zeros) and hard-decision Viterbi reference decoding.
- **Reed-Solomon (`rs_255_223`, `rs_255_239`, `rs_64_48`)**: Multi-block byte segmentation over GF($2^8$) with Berlekamp-Massey error correction validation.
- **Concatenated (`concat_rs255223_conv_k7`, `concat_rs255239_conv_k3`)**: Outer RS $\to$ Inner Convolutional with intermediate stage ground-truth retention.
- **LDPC (`ldpc_n128_k64_r12`, `ldpc_n256_k128_r12`, `ldpc_n512_k256_r12`, `ldpc_n96_k64_r23`)**: Systematic Quasi-Cyclic LDPC codes verifying $H \cdot c^T = \mathbf{0} \pmod 2$.

---

### Synthetic Engine 4 — Interleaver / Bit-Reordering Generator
Applies bit permutations to `FECRecord.encoded_bits` and produces `InterleaverRecord`:
- **None (`none`)**: Identity pass-through ($\pi(i) = i$).
- **Block (`block_8x8`, `block_16x16`, `block_16x32`, `block_32x32`)**: Matrix row-write / column-read with padding.
- **Convolutional (`conv_b4_d2`, `conv_b8_d4`)**: Ramsey/Forney $B$ delay branches with zero-flush tail and complementary delay lines.
- **Diagonal (`diag_8x8`, `diag_16x16`)**: Top-left to bottom-right anti-diagonal matrix reading ($d = r+c$).
- **Pseudo-Random (`pr_256`, `pr_512`, `pr_1024`)**: Seeded reproducible permutations with inverse mappings.

---

### Synthetic Engine 5 — Modulation / Clean IQ Waveform Generator
Translates discrete coding-layer bitstreams (`InterleaverRecord.interleaved_bits`) into pristine complex baseband IQ waveforms ($np.complex64$):
- **FSK**: 2-FSK, 4-FSK, MSK (continuous-phase direct frequency synthesis).
- **PSK**: BPSK, QPSK, 8PSK, DQPSK (Gray-coded constellation mapping + analytical RRC filtering).
- **QAM**: 16-QAM, 64-QAM, 256-QAM (2D Gray-coded grid mapping + RRC pulse shaping).
- **Power Normalization**: Strictly standardized average power ($P_{\text{avg}} = 1.0 \pm 0.05$).

---

### Synthetic Engine 6 — RF / Channel Impairment Generator
Consumes `ModulationRecord.clean_iq` waveforms and applies controlled, realistic RF impairments:
- **Canonical Execution Sequence**: Gain Scaling $\to$ Fading (Rayleigh/Rician) $\to$ Multipath FIR $\to$ Sample Clock Offset (ppm) $\to$ Timing Offset (Windowed Sinc) $\to$ CFO $\to$ Phase Offset $\to$ Frequency Drift $\to$ Interference $\to$ AWGN.
- **Exact Ground Truth**: Full stage-by-stage power telemetry, exact numerical impairment values, and source-chain lineage retention.

---

### Synthetic Engine 7 — Sampling / Capture / IQ-WAV File Generator
Serializes `ChannelRecord.impaired_iq` arrays into realistic receiver recording files:
- **Raw Binary IQ (`.iq`, `.bin`)**: `float32`, `float64`, `int16`, `int8`, little/big-endian, IQ/QI interleaving.
- **Stereo WAV (`.wav`)**: 16-bit PCM integer and 32-bit float with sample rates encoded in RIFF headers.
- **SigMF Datasets (`.sigmf-data` + `.sigmf-meta`)**: Standards-compliant JSON schema.
- **Quantization & Clipping**: Full-scale peak, target RMS, and fixed scale quantization with SNR and clipping counters.

---

## 3. Synthetic Engine 8 — Ground-Truth / Dataset Orchestrator

### Key Capabilities & Design Rules

1. **Master Seed Hierarchy**:
   - Uses `numpy.random.SeedSequence` to derive independent, deterministic child seeds for each chain index and stage (`payload_seed`, `frame_seed`, `fec_seed`, `interleaver_seed`, `modulation_seed`, `channel_seed`, `capture_seed`).
   - Ensures exact bit-level reproducibility regardless of batch size or execution order.

2. **Immutable Source Chain ID (`source_chain_id`)**:
   - Every synthetic signal transmission is assigned an immutable `source_chain_id` (e.g., `chain_00000001`).
   - Connects `payload_id`, `frame_id`, `fec_record_id`, `interleaver_record_id`, `modulation_record_id`, `channel_record_id`, and `capture_record_id`.

3. **Zero-Leakage Source-Level Splitting**:
   - Dataset splitting (default 70% Train, 15% Validation, 15% Test) occurs strictly at the `source_chain_id` level using deterministic cryptographic hashing.
   - Guaranteed: No derivatives of the same source chain can ever cross partition boundaries.

4. **Independent Parameter Sampling**:
   - Avoids shortcut correlations (e.g. all QPSK at low SNR, or all LDPC with 64QAM).
   - Validates multi-dimensional class distributions with round-robin or probability-weighted balancing.

5. **Complete Hidden Truth vs Sanitized Visible Metadata**:
   - **Visible Metadata (`<capture>.capture.json`)**: Contains only container-level attributes (file format, dtype, endianness, channels, visible sample rate). Hides all modulation, FEC, interleaver, and channel ground truth.
   - **Hidden Truth (`<capture>.truth.json`)**: Full stage-by-stage audit document containing exact original payload SHA-256, frame boundaries, coding parameters, modulation order, true SNR, CFO, fading channel taps, and file hashes.

6. **Task-Specific Deep Learning Manifests**:
   - `manifests/all.csv`: Master dataset catalog.
   - `manifests/train.csv`, `validation.csv`, `test.csv`: Partition-specific manifests.
   - `manifests/modulation.csv`: Tailored for 1D ResNet / Transformer modulation classification.
   - `manifests/fec.csv`: Tailored for blind FEC classification models.
   - `manifests/interleaver.csv`: Tailored for interleaver identification.
   - `manifests/bitstream.csv`: Tailored for frame sync and boundary segmentation models.

7. **Automated Clean-Control End-to-End Demodulation Validation**:
   - Clean/no-impairment control samples undergo reference demodulation, deinterleaving, FEC decoding, and frame extraction to confirm bit-identical recovery against the ground-truth payload.

8. **Blind Evaluation & Secret Truth Packages**:
   - `export_blind_test_package()`: Exports a sanitized evaluation dataset without truth labels to test the blind ASTRA receiver pipeline.
   - `export_evaluation_truth_package()`: Exports a corresponding secret ground-truth scoring directory.

---

## 4. Dataset Directory Layout

```
datasets/astra_synthetic_v1/
│
├── captures/                      # Physical .iq / .wav capture files
│   ├── cap_000001.iq
│   └── ...
│
├── visible_metadata/              # Sanitized container metadata (.capture.json)
│   ├── cap_000001.capture.json
│   └── ...
│
├── truth/                         # Complete hidden ground-truth documents (.truth.json)
│   ├── cap_000001.truth.json
│   └── ...
│
├── manifests/                     # Master, split, and task manifests
│   ├── all.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── modulation.csv
│   ├── fec.csv
│   ├── interleaver.csv
│   └── bitstream.csv
│
├── reports/                       # Validation, quality, and distribution audit reports
│   ├── data_quality.json
│   ├── leakage_check.json
│   └── distribution_report.json
│
└── dataset_metadata.json          # Global dataset summary and configuration SHA-256
```

---

## 5. Python API Usage

### End-to-End Dataset Generation with Orchestrator

```python
from astra_synthetic.orchestration import ASTRASyntheticDatasetGenerator

# 1. Initialize Generator with YAML configuration or defaults
generator = ASTRASyntheticDatasetGenerator(config="astra_synthetic/configs/dataset_config.yaml")

# 2. Stream and generate 100 synthetic signal chains
records = generator.generate_dataset(
    count=100,
    output_root="./datasets/astra_synthetic_v1",
    show_progress=True,
)

# 3. Export sanitized blind test set for receiver evaluation
blind_pkg = generator.export_blind_test_set(output_dir="./datasets/blind_test")

# 4. Export secret evaluation truth package for scoring
eval_truth_pkg = generator.export_evaluation_truth(output_dir="./datasets/evaluation_truth")
```

---

## 6. Running Tests & Demonstrations

### Execute Full Pytest Test Suite
```bash
pytest astra_synthetic/tests -v
```

### Run Demonstration Scripts
```bash
# Engine 1 (Payloads)
python -m astra_synthetic.examples.generate_payloads

# Engine 2 (Frames)
python -m astra_synthetic.examples.generate_frames

# Engine 3 (FEC Encoding)
python -m astra_synthetic.examples.generate_fec_examples

# Engine 4 (Interleaver / Bit-Reordering)
python -m astra_synthetic.examples.generate_interleaver_examples

# Engine 5 (Modulation / Clean IQ Waveforms)
python -m astra_synthetic.examples.generate_modulation_examples

# Engine 6 (RF / Channel Impairments)
python -m astra_synthetic.examples.generate_channel_examples

# Engine 7 (Capture / IQ-WAV File Generation)
python -m astra_synthetic.examples.generate_capture_examples

# Engine 8 (End-to-End Dataset Orchestration & Manifests)
python -m astra_synthetic.examples.generate_full_dataset
```
