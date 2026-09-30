"""
ASTRA Doctor: Comprehensive Diagnostic & System Health Utility.
Checks Python version, core dependencies, GPU/CUDA, OpenGL capabilities,
model checkpoints, configuration files, and write permissions.
Outputs SYSTEM_HEALTH_REPORT.json.
"""

import sys
import os
import json
import shutil
import platform
from pathlib import Path
from typing import Dict, Any


def run_astra_doctor(output_json: str = "SYSTEM_HEALTH_REPORT.json") -> Dict[str, Any]:
    """Runs a full suite of system health diagnostics and writes report to disk."""
    report: Dict[str, Any] = {
        "status": "HEALTHY",
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": {},
        "hardware": {},
        "configs": {},
        "models": {},
        "filesystem": {},
        "issues": []
    }

    # 1. Dependency checks
    packages = [
        ("numpy", "2.0.0"),
        ("scipy", "1.15.0"),
        ("torch", "2.0.0"),
        ("sklearn", "1.5.0"),
        ("xgboost", "2.0.0"),
        ("joblib", "1.4.0"),
        ("reedsolo", "1.7.0"),
        ("PySide6", "6.8.0"),
        ("pyqtgraph", "0.13.0"),
        ("OpenGL", "3.1.0"),
        ("yaml", "6.0")
    ]

    for pkg_name, min_ver in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            report["dependencies"][pkg_name] = {"installed": True, "version": ver}
        except ImportError as e:
            report["dependencies"][pkg_name] = {"installed": False, "error": str(e)}
            report["issues"].append(f"Missing dependency: {pkg_name}")
            report["status"] = "DEGRADED"

    # 2. Hardware & Acceleration
    try:
        import torch
        report["hardware"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["hardware"]["device_count"] = torch.cuda.device_count()
            report["hardware"]["device_name"] = torch.cuda.get_device_name(0)
            report["hardware"]["cuda_version"] = torch.version.cuda
        else:
            report["hardware"]["active_device"] = "CPU (Fallback)"
    except Exception as e:
        report["hardware"]["cuda_error"] = str(e)

    # 3. OpenGL check
    try:
        import OpenGL.GL as gl
        report["hardware"]["opengl_installed"] = True
    except Exception as e:
        report["hardware"]["opengl_installed"] = False
        report["hardware"]["opengl_error"] = str(e)
        report["issues"].append("PyOpenGL not functional; 3D viewport will fallback to 2D")

    # 4. Configuration check
    base_dir = Path(__file__).resolve().parent.parent
    config_dir = base_dir / "configs"
    if config_dir.is_dir():
        from astra_config.loader import validate_config_dir
        val_res = validate_config_dir(config_dir)
        report["configs"] = val_res
        if not val_res["valid"]:
            report["issues"].extend(val_res["errors"])
            report["issues"].extend([f"Missing config: {m}" for m in val_res["missing_files"]])
            report["status"] = "DEGRADED"
    else:
        report["configs"] = {"valid": False, "error": "configs/ directory missing"}
        report["issues"].append("Missing configs directory")
        report["status"] = "CRITICAL"

    # 5. Model Checkpoint Verification
    checkpoints = [
        ("resnet1d_v2", base_dir / "checkpoints" / "astra_resnet1d_modulation_v2.pt"),
        ("cnn2d_v2", base_dir / "checkpoints" / "astra_spectrogram_cnn_v2.pt"),
        ("resnet1d_legacy", base_dir / "best_model_resnet1d.pt"),
        ("cnn2d_legacy", base_dir / "outputs" / "CSPB.ML.2018R2" / "best_model.pt"),
        ("symbol_rate_ranker", base_dir / "checkpoints" / "symbol_rate_ranker.joblib"),
        ("random_forest", base_dir / "checkpoints" / "random_forest_support.joblib"),
        ("pipeline_scorer_xgb", base_dir / "checkpoints" / "pipeline_scorer_xgb" / "xgboost_model.json"),
    ]

    for model_name, path in checkpoints:
        exists = path.is_file()
        size_bytes = path.stat().st_size if exists else 0
        report["models"][model_name] = {
            "path": str(path),
            "exists": exists,
            "size_mb": round(size_bytes / (1024 * 1024), 2) if exists else 0.0
        }
        if not exists:
            report["issues"].append(f"Model checkpoint missing: {model_name} at {path}")
            if report["status"] != "CRITICAL":
                report["status"] = "DEGRADED"

    # 6. Filesystem & Write Permissions
    for dir_name in ["outputs", "logs", ".cache"]:
        d_path = base_dir / dir_name
        try:
            d_path.mkdir(parents=True, exist_ok=True)
            test_file = d_path / ".perm_check"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            report["filesystem"][dir_name] = {"writable": True}
        except Exception as e:
            report["filesystem"][dir_name] = {"writable": False, "error": str(e)}
            report["issues"].append(f"Directory not writable: {dir_name}")
            report["status"] = "CRITICAL"

    if report["issues"] and report["status"] == "HEALTHY":
        report["status"] = "DEGRADED"

    # Save to disk
    out_path = Path(output_json)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save {output_json}: {e}", file=sys.stderr)

    return report
