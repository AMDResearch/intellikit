#!/usr/bin/env python3
"""
validate.py — Derived counter validation runner for metrix.

Compiles microbenchmarks, profiles them with the metrix API, and checks
that derived metric values match analytically expected results.

This validates the YAML expressions in counter_defs.yaml, NOT raw
hardware counters.  The goal: given a kernel with known behavior,
does metrix report the correct derived value?

Usage:
    python validate.py                          # run all benchmarks
    python validate.py --benchmark bw_copy      # run one benchmark
    python validate.py --arch gfx942            # explicit arch
    python validate.py --list                   # list available benchmarks
    python validate.py --output results.json    # save results to file
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add metrix to path if running from source tree
BENCH_DIR = Path(__file__).resolve().parent
METRIX_SRC = BENCH_DIR.parent / "src"
if METRIX_SRC.is_dir():
    sys.path.insert(0, str(METRIX_SRC))

from expectations import EXPECTATIONS, check_expectation, get_benchmarks_for_arch


def compile_benchmarks(bench_dir: Path) -> bool:
    """Compile all benchmark kernels via make."""
    print("=" * 60)
    print("COMPILING BENCHMARKS")
    print("=" * 60)
    result = subprocess.run(
        ["make", "-j", "4"],
        cwd=str(bench_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"COMPILE FAILED:\n{result.stderr}")
        return False
    print("Compilation successful.")
    return True


def detect_arch() -> str:
    """Auto-detect GPU architecture via metrix."""
    from metrix.backends.detect import detect_gpu_arch
    return detect_gpu_arch()


def get_rocm_info() -> dict:
    """Collect ROCm environment info for the report."""
    info = {}

    # ROCm version
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if "ROCk module" in line:
                info["rock_version"] = line.strip()
            if "Name:" in line and "gfx" in line.lower():
                info["gpu_name"] = line.strip()
    except Exception:
        pass

    # rocprofv3 version
    try:
        r = subprocess.run(["rocprofv3", "--version"], capture_output=True, text=True, timeout=10)
        info["rocprofv3_version"] = r.stdout.strip() or r.stderr.strip()
    except Exception:
        pass

    # Docker image
    if os.path.exists("/.dockerenv"):
        info["in_docker"] = True

    return info


def run_benchmark(
    bench_dir: Path,
    name: str,
    spec: dict,
    arch: str,
    num_replays: int = 3,
) -> dict:
    """
    Profile one benchmark with metrix API and check derived metrics.

    Returns a result dict with pass/fail per metric.
    """
    from metrix.api import Metrix

    binary = bench_dir / "bin" / spec["binary"]
    if not binary.exists():
        return {
            "benchmark": name,
            "status": "SKIP",
            "reason": f"Binary not found: {binary}",
        }

    # Build the command with args
    cmd_parts = [str(binary)] + spec.get("args", [])
    command = " ".join(cmd_parts)
    metrics = spec["metrics"]

    print(f"\n{'─' * 60}")
    print(f"BENCHMARK: {name}")
    print(f"  Command:  {command}")
    print(f"  Metrics:  {', '.join(metrics)}")
    print(f"  Expected: {spec['description']}")

    try:
        profiler = Metrix(arch=arch)
        results = profiler.profile(
            command=command,
            metrics=metrics,
            num_replays=num_replays,
            aggregate_by_kernel=True,
            timeout_seconds=120,
        )
    except Exception as e:
        print(f"  PROFILE ERROR: {e}")
        return {
            "benchmark": name,
            "status": "ERROR",
            "reason": str(e),
        }

    if not results.kernels:
        print("  WARNING: No kernels profiled.")
        return {
            "benchmark": name,
            "status": "ERROR",
            "reason": "No kernels returned by profiler",
        }

    # Use the last kernel (measured run, not warmup) unless only one
    # Pick the kernel with the longest duration as the "main" one
    kernel = max(results.kernels, key=lambda k: k.duration_us.avg)

    print(f"  Kernel:   {kernel.name}")
    print(f"  Duration: {kernel.duration_us.avg:.1f} us (avg)")

    # Check each metric against expectations
    metric_results = {}
    all_pass = True

    for metric_name in metrics:
        if metric_name not in kernel.metrics:
            print(f"  {metric_name}: NOT REPORTED")
            metric_results[metric_name] = {
                "pass": False,
                "reason": "metric not in profiling results",
                "value": None,
            }
            all_pass = False
            continue

        value = kernel.metrics[metric_name].avg
        expected = spec.get("expected", {}).get(metric_name)

        if expected is None:
            # No expectation — just report the value
            print(f"  {metric_name}: {value:.4f} (no expectation)")
            metric_results[metric_name] = {
                "pass": True,
                "reason": "no expectation (informational)",
                "value": value,
            }
            continue

        check = check_expectation(metric_name, value, expected)
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  {metric_name}: {value:.4f} — {status} ({check['reason']})")
        metric_results[metric_name] = check

        if not check["pass"]:
            all_pass = False

    return {
        "benchmark": name,
        "status": "PASS" if all_pass else "FAIL",
        "kernel": kernel.name,
        "duration_us": kernel.duration_us.avg,
        "metrics": metric_results,
    }


def generate_report(
    results: List[dict],
    arch: str,
    rocm_info: dict,
) -> str:
    """Generate a markdown summary report."""
    lines = []
    lines.append("# Metrix Derived Counter Validation Report\n")
    lines.append(f"**Architecture:** {arch}")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for k, v in rocm_info.items():
        lines.append(f"**{k}:** {v}")
    lines.append("")

    # Summary table
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    lines.append(f"## Summary: {passed}/{total} passed, {failed} failed, "
                 f"{errors} errors, {skipped} skipped\n")

    lines.append("| Benchmark | Status | Details |")
    lines.append("|-----------|--------|---------|")
    for r in results:
        details = ""
        if r["status"] in ("ERROR", "SKIP"):
            details = r.get("reason", "")
        elif "metrics" in r:
            fails = [m for m, v in r["metrics"].items() if not v.get("pass", True)]
            if fails:
                details = "Failed: " + ", ".join(fails)
            else:
                details = f"{len(r['metrics'])} metrics checked"
        lines.append(f"| {r['benchmark']} | **{r['status']}** | {details} |")

    lines.append("")

    # Detailed results per benchmark
    lines.append("## Detailed Results\n")
    for r in results:
        if r["status"] in ("ERROR", "SKIP"):
            lines.append(f"### {r['benchmark']} — {r['status']}\n")
            lines.append(f"{r.get('reason', '')}\n")
            continue

        lines.append(f"### {r['benchmark']} — {r['status']}\n")
        if "kernel" in r:
            lines.append(f"- **Kernel:** `{r['kernel']}`")
            lines.append(f"- **Duration:** {r.get('duration_us', 0):.1f} us\n")

        if "metrics" in r:
            lines.append("| Metric | Value | Expected | Result |")
            lines.append("|--------|-------|----------|--------|")
            for metric, check in r["metrics"].items():
                val = f"{check['value']:.4f}" if check["value"] is not None else "N/A"
                exp = check.get("expected", "N/A")
                status = "PASS" if check["pass"] else "**FAIL**"
                lines.append(f"| {metric} | {val} | {exp} | {status} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate metrix derived counters")
    parser.add_argument("--arch", type=str, help="GPU architecture (auto-detect if omitted)")
    parser.add_argument("--benchmark", type=str, help="Run a single benchmark by name")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--output", type=str, help="Save JSON results to file")
    parser.add_argument("--report", type=str, help="Save markdown report to file")
    parser.add_argument("--num-replays", type=int, default=3, help="Profiling replays per benchmark")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation step")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name, spec in EXPECTATIONS.items():
            print(f"  {name:30s} — {spec['description']}")
        return 0

    # Detect arch
    arch = args.arch
    if arch is None:
        try:
            arch = detect_arch()
            print(f"Auto-detected architecture: {arch}")
        except Exception as e:
            print(f"Could not auto-detect GPU arch: {e}")
            print("Pass --arch explicitly.")
            return 1

    # Compile
    if not args.skip_compile:
        if not compile_benchmarks(BENCH_DIR):
            return 1

    # Select benchmarks
    benchmarks = get_benchmarks_for_arch(arch)
    if args.benchmark:
        if args.benchmark in benchmarks:
            benchmarks = {args.benchmark: benchmarks[args.benchmark]}
        else:
            print(f"Unknown benchmark: {args.benchmark}")
            print(f"Available for {arch}: {', '.join(benchmarks.keys())}")
            return 1

    # Collect ROCm info
    rocm_info = get_rocm_info()

    # Run benchmarks
    print(f"\nRunning {len(benchmarks)} benchmarks on {arch}")
    print(f"Replays per benchmark: {args.num_replays}")

    all_results = []
    for name, spec in benchmarks.items():
        result = run_benchmark(
            bench_dir=BENCH_DIR,
            name=name,
            spec=spec,
            arch=arch,
            num_replays=args.num_replays,
        )
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    errors = sum(1 for r in all_results if r["status"] == "ERROR")
    total = len(all_results)
    print(f"  PASS:  {passed}/{total}")
    print(f"  FAIL:  {failed}/{total}")
    print(f"  ERROR: {errors}/{total}")

    for r in all_results:
        if r["status"] == "FAIL" and "metrics" in r:
            for metric, check in r["metrics"].items():
                if not check.get("pass", True):
                    print(f"  FAIL: {r['benchmark']}.{metric} — {check['reason']}")

    # Save outputs
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"arch": arch, "rocm_info": rocm_info, "results": all_results}, f, indent=2)
        print(f"\nJSON results saved to: {args.output}")

    report = generate_report(all_results, arch, rocm_info)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report)
        print(f"Markdown report saved to: {args.report}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
