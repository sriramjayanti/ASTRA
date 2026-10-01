"""
ASTRA Production Unified Command Line Interface (CLI).
Provides developer, researcher, and headless pipeline entrypoints.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Ensure root workspace is accessible
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from astra_config.loader import load_astra_config, validate_config_dir
from astra_cli.doctor import run_astra_doctor
from astra_cli.analyze import analyze_signal


def cmd_doctor(args: argparse.Namespace) -> int:
    """Executes astra doctor diagnostic health check."""
    print("=" * 70)
    print("          ASTRA SYSTEM HEALTH CHECK & DOCTOR")
    print("=" * 70)
    report = run_astra_doctor(output_json=args.output or "SYSTEM_HEALTH_REPORT.json")

    print(f"Overall Status: {report['status']}")
    print(f"Python:         {report['system']['python_version'].split()[0]}")
    print(f"Platform:       {report['system']['platform']}")

    print("\n[HARDWARE ACCELERATION]")
    hw = report.get("hardware", {})
    cuda_avail = hw.get("cuda_available", False)
    print(f"  CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"  GPU Name:       {hw.get('device_name')}")
    else:
        print(f"  Backend:        {hw.get('active_device', 'CPU')}")

    print("\n[AI MODELS & CHECKPOINTS]")
    for mname, mdata in report.get("models", {}).items():
        st = "FOUND" if mdata.get("exists") else "MISSING"
        print(f"  - {mname:22s} [{st:7s}] {mdata.get('size_mb', 0)} MB")

    print("\n[CONFIGURATIONS]")
    cfg_data = report.get("configs", {})
    print(f"  Valid: {cfg_data.get('valid')}, Checked: {cfg_data.get('files_checked', 0)} YAML files")

    if report.get("issues"):
        print("\n[WARNINGS / ISSUES]")
        for issue in report["issues"]:
            print(f"  ! {issue}")

    print(f"\nDetailed JSON health report written to: {args.output or 'SYSTEM_HEALTH_REPORT.json'}")
    return 0 if report["status"] in ["HEALTHY", "DEGRADED"] else 1


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Validates configuration files."""
    config_dir = Path(args.dir) if args.dir else ROOT_DIR / "configs"
    print(f"Validating configuration directory: {config_dir}")
    res = validate_config_dir(config_dir)
    if res["valid"]:
        print(f"[SUCCESS] All {res['files_checked']} configuration files are present and valid.")
        return 0
    else:
        print("[ERROR] Configuration validation failed:")
        for missing in res.get("missing_files", []):
            print(f"  - Missing file: {missing}")
        for err in res.get("errors", []):
            print(f"  - Error: {err}")
        return 1


def cmd_models(args: argparse.Namespace) -> int:
    """Inspects all ASTRA model checkpoints and registries."""
    print("=" * 70)
    print("              ASTRA REGISTERED AI MODELS")
    print("=" * 70)
    checkpoints = [
        ("ResNet-1D V2 (10-Class Time)", ROOT_DIR / "checkpoints" / "astra_resnet1d_v2.pt"),
        ("CNN-2D V2 (10-Class Spectrogram)", ROOT_DIR / "checkpoints" / "astra_spectrogram_cnn_v2.pt"),
        ("Symbol Rate Ranker", ROOT_DIR / "checkpoints" / "symbol_rate_ranker.joblib"),
        ("Constellation RF Support", ROOT_DIR / "checkpoints" / "random_forest_support.joblib"),
        ("Pipeline Scorer (XGBoost)", ROOT_DIR / "checkpoints" / "pipeline_scorer_xgb" / "xgboost_model.json"),
    ]
    for name, p in checkpoints:
        exists = p.exists()
        size_mb = round(p.stat().st_size / (1024 * 1024), 2) if exists else 0.0
        status = "AVAILABLE" if exists else "NOT FOUND"
        print(f"  * {name:28s} [{status:9s}] {size_mb:6.2f} MB | {p.name}")
    return 0


def cmd_system_info(args: argparse.Namespace) -> int:
    """Displays platform and runtime architecture details."""
    import platform
    import torch
    print("=" * 70)
    print("                 ASTRA SYSTEM INFORMATION")
    print("=" * 70)
    print(f"  ASTRA Version:     1.0.0 (Production Hardened)")
    print(f"  OS / Kernel:       {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  Python Runtime:    {sys.version.split()[0]} ({sys.executable})")
    print(f"  PyTorch Version:   {torch.__version__}")
    print(f"  CUDA Available:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Active GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Arch:         {torch.cuda.get_device_capability(0)}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Executes headless analysis on a raw IQ capture."""
    print(f"Analyzing signal capture: {args.iq_file} (Sample Rate: {args.sample_rate} Hz)...")
    try:
        results = analyze_signal(
            file_path=args.iq_file,
            sample_rate=args.sample_rate,
            config_dir=args.config_dir,
            output_format=args.format
        )
        if args.format == "json":
            print(json.dumps(results, indent=2))
        else:
            print("\n" + "=" * 60)
            print("                 ANALYSIS RESULTS")
            print("=" * 60)
            print(f"  Source File:        {results.get('source_file')}")
            print(f"  Samples Analyzed:   {results.get('sample_count')} ({results.get('duration_ms')} ms)")
            print(f"  Signal Power:       {results.get('power_dbfs')} dBFS")
            mod_stage = results.get("stages", {}).get("modulation", {})
            print(f"  Modulation:         {mod_stage.get('predicted_class')} (Conf: {mod_stage.get('confidence', 0):.2%})")
            sr_stage = results.get("stages", {}).get("symbol_rate", {})
            print(f"  Symbol Rate:        {sr_stage.get('estimated_baud')} Baud (SPS: {sr_stage.get('sps')})")
            summary = results.get("summary", {})
            print(f"  Status:             {summary.get('overall_status')} (Confidence: {summary.get('overall_confidence', 0):.2%})")
            print("=" * 60)
        return 0
    except Exception as e:
        print(f"[ERROR] Signal analysis failed: {str(e)}", file=sys.stderr)
        return 1


def cmd_run_demo(args: argparse.Namespace) -> int:
    """Runs standard synthetic demo capture through complete AI pipeline."""
    import run_complete_astra_ai
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astra",
        description="ASTRA: Automated Signal Analysis & Recovery Assistant CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Run system diagnostics and generate health report")
    p_doc.add_argument("--output", "-o", default="SYSTEM_HEALTH_REPORT.json", help="Path to output JSON")

    # validate-config
    p_val = subparsers.add_parser("validate-config", help="Validate YAML configuration files")
    p_val.add_argument("--dir", "-d", default=None, help="Directory containing config YAMLs")

    # models
    subparsers.add_parser("models", help="Inspect model checkpoints and status")

    # system-info
    subparsers.add_parser("system-info", help="Display platform, Python, GPU, and CUDA information")

    # analyze
    p_ana = subparsers.add_parser("analyze", help="Headless IQ capture analysis")
    p_ana.add_argument("iq_file", help="Path to IQ file (.iq, .bin, .npy)")
    p_ana.add_argument("--sample-rate", "-fs", type=float, default=192000.0, help="Sampling frequency in Hz")
    p_ana.add_argument("--config-dir", "-c", default=None, help="Path to configs directory")
    p_ana.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")

    # run-demo
    subparsers.add_parser("run-demo", help="Run synthetic demo signal end-to-end")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "doctor": cmd_doctor,
        "validate-config": cmd_validate_config,
        "models": cmd_models,
        "system-info": cmd_system_info,
        "analyze": cmd_analyze,
        "run-demo": cmd_run_demo,
    }

    ret = dispatch[args.command](args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
