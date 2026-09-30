# ASTRA Stage 14 — Header / Payload Explorer

`astra_payload_explorer` is the Stage 14 module of the **ASTRA (Automated Signal Analysis & Recovery Assistant)** pipeline. It receives the recovered bitstream candidate together with Stage 12 (Bitstream Intelligence) features and Stage 13 (1D CNN + Transformer) structural predictions, and segments, parses, interprets, and exports candidate protocol fields and payload content.

---

## 1. Primary Capabilities

1. **Dual Operating Modes**:
   - **Profile-Aware Parsing**: Parses known protocols using structured YAML schemas (offsets, bit-widths, typed encodings, enums, bitfields, fixed-point units).
   - **Blind Exploratory Parsing**: For unknown protocols, identifies constant fields, counter candidates, length candidates, and variable regions without inventing false semantics.
   - **Auto Mode**: Automatically checks registry profiles, computes compatibility scores, and falls back to blind exploration if confidence is insufficient.

2. **Frame Resolution & Segmentation**:
   - Synthesizes evidence from Stage 12 periodicity/sync candidates, Stage 13 transformer boundary spans, and known profile schemas.
   - Isolates `SYNC`, `HEADER`, `PAYLOAD`, `CRC`, `PADDING`, and `UNKNOWN` regions.
   - Preserves partial leading/trailing frames and damaged (CRC fail) frames with explicit quality flags.

3. **Multi-View Payload Extraction**:
   - **Non-destructive**: Never alters or discards raw bitstream content.
   - Provides parallel views: `RAW_BITS`, `HEX`, `BYTES`, `UINT8 ARRAY`, `UTF-8`, `ASCII`, `BASE64`, and file magic signatures (PNG, JPEG, PDF, ZIP, GZIP, ELF).
   - Fully supports non-byte-aligned payload bit lengths without truncating trailing bits.

4. **Cross-Frame Analysis**:
   - Detects duplicate frames, identical payloads, and sequence continuity across multiple frames.
   - Analyzes payload variations and Hamming distance progressions.

5. **Confidence & Interpretation Provenance**:
   - Every parsed field and interpretation is explicitly categorized:
     - `KNOWN`: Backed by an explicit protocol profile.
     - `INFERRED`: Backed by strong statistical or cross-frame evidence (e.g. strict +1 sequence increment).
     - `POSSIBLE`: Heuristic or candidate hypothesis.
     - `UNKNOWN`: Unexplained bits/fields.
   - Maintains separate scores for boundary confidence, content representation validity, and frame quality.

---

## 2. Directory Structure

```
astra_payload_explorer/
│
├── configs/
│   ├── payload_explorer_config.yaml  # Top-level explorer settings
│   ├── protocol_profiles.yaml        # Known protocol schemas (e.g. astra_demo_v1, ccsds_telemetry_packet)
│   └── payload_decoders.yaml         # File magic signatures and text decoding configs
│
├── src/
│   ├── __init__.py                   # Package exports
│   ├── models.py                     # Dataclasses (FrameRecord, ParsedField, PayloadViews, PayloadExplorerResult)
│   ├── frame_resolver.py             # Multi-source frame hypothesis resolver
│   ├── segmentation.py               # Stream-to-frames slicer and SHA-256 bit hashing
│   ├── header_parser.py              # Profile-aware typed header field parser
│   ├── blind_fields.py               # Blind constant, counter, and length field detector
│   ├── payload_extractor.py          # Frame payload isolator
│   ├── payload_decoders.py           # Multi-representation payload decoder
│   ├── byte_alignment.py             # Bit/byte conversion and hex formatting utilities
│   ├── bit_order.py                  # MSB/LSB bit order and integer two's-complement
│   ├── endian.py                     # Big/little-endian integer and fixed-point decoders
│   ├── cross_frame.py                # Cross-frame duplicates and sequence continuity
│   ├── confidence.py                 # Multi-dimensional confidence engine
│   ├── exporters.py                  # JSON, CSV, and canonical hex dump exporter
│   ├── validators.py                 # Protocol profile matcher and compatibility scorer
│   ├── inference.py                  # Main HeaderPayloadExplorer orchestrator
│   └── utils.py                      # Synthetic stream and framed payload generator
│
├── tests/
│   ├── test_endian.py                # Bit/byte conversions, two's-complement, fixed-point
│   ├── test_header_parser.py         # Profile-aware field decoders and enums
│   ├── test_frame_resolver.py        # Frame hypothesis resolution and disagreement handling
│   ├── test_segmentation.py          # Slicing, partial frame handling, and hashing
│   ├── test_payload_decoders.py      # UTF-8, ASCII, binary preservation, magic signatures
│   ├── test_blind_fields.py          # Constant, counter, and length candidate discovery
│   ├── test_cross_frame.py           # Duplicate detection and JSON serialization
│   └── test_end_to_end_explorer.py   # Multi-protocol matching, blind fallback, and synthetic tests
│
├── examples/
│   └── explore_payload.py            # Complete end-to-end demonstration script
│
├── requirements.txt
├── run_tests.py
└── README.md
```

---

## 3. Quick Start

### Running Tests
```bash
python astra_payload_explorer/run_tests.py
```

### Running the Example
```bash
python astra_payload_explorer/examples/explore_payload.py
```

---

## 4. Python API Example

```python
import numpy as np
from astra_payload_explorer.src.inference import HeaderPayloadExplorer
from astra_payload_explorer.src.models import PayloadExplorerContext

# 1. Recovered hard bits from Stage 11 / Stage 10
recovered_bits = np.array([...], dtype=np.uint8)

# 2. Context with Stage 12 & Stage 13 structural evidence
ctx = PayloadExplorerContext(
    pipeline_path_id="best_pipe_001",
    stage12_result=stage12_output,
    stage13_result=stage13_output,
    mode="auto"
)

# 3. Explore
explorer = HeaderPayloadExplorer()
result = explorer.explore(recovered_bits, ctx)

print(f"Status: {result.explorer_status.value}")
print(f"Frames: {result.frame_count}")
for frame in result.frames:
    print(f"Frame #{frame.frame_index}: Payload UTF-8 = {frame.payload.representations.get('utf8', {}).get('value')}")
```
