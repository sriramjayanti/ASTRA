"""
ASTRA Installation and Environment Health Validation Script
Verifies Python dependencies, PyTorch CUDA capability, models, configs, and core modules.
"""

import sys
import os
import json
from pathlib import Path

def validate_environment():
    print("=" * 60)
    print("  ASTRA SYSTEM HEALTH & INSTALLATION VALIDATION")
    print("=" * 60)
    
    # 1. Python Version Check
    py_ver = sys.version.split()[0]
    print(f"[*] Python Version: {py_ver}")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print(f"[-] WARNING: ASTRA recommends Python 3.9+. Found {py_ver}")
    else:
        print(f"[+] Python version compatible: PASS")

    # 2. Dependency Imports
    deps = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("torch", "PyTorch"),
        ("xgboost", "XGBoost"),
        ("sklearn", "scikit-learn"),
        ("joblib", "Joblib"),
        ("yaml", "PyYAML"),
    ]
    
    print("\n[*] Checking Core Scientific Dependencies...")
    for mod_name, label in deps:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [+] {label:<15}: {ver} (OK)")
        except ImportError as e:
            print(f"  [-] {label:<15}: MISSING ({e})")

    # 3. PyTorch & CUDA Check
    print("\n[*] Checking PyTorch Compute Hardware...")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            print(f"  [+] CUDA Available: YES (Device: {device_name})")
        else:
            print(f"  [i] CUDA Available: NO (CPU fallback active)")
    except Exception as e:
        print(f"  [-] PyTorch check error: {e}")

    # 4. Optional GUI Dependencies Check
    print("\n[*] Checking GUI Dependencies (Optional for CLI/Headless)...")
    gui_mods = [("PyQt6", "PyQt6"), ("PySide6", "PySide6"), ("pyqtgraph", "PyQtGraph")]
    for mod_name, label in gui_mods:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "available")
            print(f"  [+] {label:<15}: {ver} (OK)")
        except ImportError:
            print(f"  [i] {label:<15}: Not installed (GUI mode requires PyQt6/PySide6)")

    # 5. Model Checkpoints & Manifest Validation
    print("\n[*] Validating Model Registry & Checkpoints...")
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / "models" / "model_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            models = manifest.get("models", {})
            print(f"  [+] Manifest loaded: models/model_manifest.json ({len(models)} registered models)")
            for m_id, m_info in models.items():
                rel_path = m_info.get("checkpoint_path", "")
                full_path = repo_root / rel_path
                exists = full_path.exists()
                status = "FOUND" if exists else "MISSING"
                print(f"    - {m_id:<28}: {status} ({rel_path})")
        except Exception as e:
            print(f"  [-] Manifest parsing error: {e}")
    else:
        print(f"  [-] Model manifest missing at {manifest_path}")

    # 6. Core Modules Import Validation
    print("\n[*] Validating ASTRA Pipeline Architecture Imports...")
    astra_packages = [
        "astra_synthetic",
        "astra_modulation_v2",
        "astra_symbol_rate",
        "astra_synchronization",
        "astra_demodulation",
        "astra_interleaver",
        "astra_fec",
        "astra_validation",
        "astra_bitstream_intelligence",
        "astra_explainability",
        "astra_candidate_engine",
    ]
    sys.path.insert(0, str(repo_root))
    all_pkg_ok = True
    for pkg in astra_packages:
        try:
            __import__(pkg)
            print(f"  [+] {pkg:<32}: OK")
        except Exception as e:
            print(f"  [-] {pkg:<32}: FAILED ({e})")
            all_pkg_ok = False

    print("\n" + "=" * 60)
    if all_pkg_ok:
        print("  ASTRA SYSTEM HEALTH CHECK: PASSED (System Ready)")
    else:
        print("  ASTRA SYSTEM HEALTH CHECK: COMPLETED WITH WARNINGS")
    print("=" * 60)

if __name__ == "__main__":
    validate_environment()
