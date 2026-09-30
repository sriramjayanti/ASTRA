"""
ASTRA Synthetic Engine 1: Payload / Bitstream Generator Example Script.
Demonstrates generation, inspection, and multi-format serialization of payloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("astra_example")

# Import ASTRA payload components
import sys
# Add parent directory of astra_synthetic if running directly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astra_synthetic.payload import (
    PayloadGenerator,
    save_payload,
    save_batch,
)


def run_demonstration(output_dir: Path | str = "output/demo_payloads") -> list:
    """Generate and display the 5 core milestone demonstration payloads."""
    config_path = current_dir.parent / "configs" / "payload_config.yaml"
    generator = PayloadGenerator(config_path if config_path.exists() else None)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ASTRA SYNTHETIC ENGINE 1 - DEMONSTRATION PAYLOAD GENERATION")
    print("=" * 80 + "\n")
    
    records = []

    # 1. 1024-bit random binary payload
    p1 = generator.generate(
        payload_type="random_bits",
        bit_length=1024,
        seed=1001,
        payload_id="payload_000001",
    )
    records.append(p1)

    # 2. "ASTRA TEST" text payload
    p2 = generator.generate(
        payload_type="text",
        text="ASTRA TEST",
        payload_id="payload_000002",
    )
    records.append(p2)

    # 3. 2048-bit repeated 10110011 payload
    p3 = generator.generate(
        payload_type="repeated_pattern",
        pattern="10110011",
        bit_length=2048,
        payload_id="payload_000003",
    )
    records.append(p3)

    # 4. 512-byte counter payload
    p4 = generator.generate(
        payload_type="counter",
        byte_length=512,
        start_value=0,
        payload_id="payload_000004",
    )
    records.append(p4)

    # 5. Biased random payload with P(1) = 0.10
    p5 = generator.generate(
        payload_type="biased_random",
        bit_length=1024,
        probability_one=0.10,
        seed=1005,
        payload_id="payload_000005",
    )
    records.append(p5)

    # Display and save each payload
    for i, rec in enumerate(records, 1):
        first_64_bits = "".join(str(b) for b in rec.payload_bits[:64])
        saved_dir = save_payload(rec, output_dir=out_path)
        
        print(f"[{i}] Payload ID:        {rec.payload_id}")
        print(f"    Type:              {rec.payload_type}")
        print(f"    Bit Length:        {rec.bit_length} bits ({rec.byte_length} bytes)")
        print(f"    First 64 Bits:     {first_64_bits}")
        print(f"    Shannon Entropy:   {rec.entropy_estimate:.6f}")
        print(f"    SHA-256 Hash:      {rec.sha256}")
        print(f"    Saved Directory:   {saved_dir}")
        print("-" * 80)

    print("\nDemonstration payloads saved successfully to:", out_path.resolve())
    return records


def run_batch_demonstration(count: int = 100, output_dir: Path | str = "output/batch_demo"):
    """Demonstrate streaming batch generation with mixed distribution."""
    config_path = current_dir.parent / "configs" / "payload_config.yaml"
    generator = PayloadGenerator(config_path if config_path.exists() else None)
    
    print(f"\nStreaming {count} mixed-type payloads using iter_payloads()...")
    batch_records = []
    for rec in generator.iter_payloads(count):
        batch_records.append(rec)
    
    print(f"Generated {len(batch_records)} payloads in memory.")
    
    # Save first 10 to disk as demonstration
    sample_dir = Path(output_dir)
    save_batch(batch_records[:10], output_dir=sample_dir)
    print(f"Saved first 10 sample payloads to {sample_dir.resolve()}")


if __name__ == "__main__":
    run_demonstration()
    run_batch_demonstration(count=50)
