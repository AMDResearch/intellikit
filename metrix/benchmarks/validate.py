#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Main validation runner for metrix derived counter validation.

Runs each microbenchmark through the metrix API, collects derived metrics,
compares against analytically expected values, and produces a report.

Usage:
    python validate.py [--arch ARCH] [--bench NAME] [--num-replays N]
                       [--output FILE] [--verbose] [--raw-counters]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add metrix to path
SCRIPT_DIR = Path(__file__).parent.resolve()
METRIX_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(METRIX_DIR / "src"))

from metrix import Metrix
from expectations import get_expectations, BenchmarkExpectation, Expectation


def detect_arch() -> str:
    """Detect GPU architecture from rocminfo."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if "Name:" in line and "gfx" in line:
                return line.split()[-1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "gfx942"


def check_binary_exists(binary: str) -> bool:
    """Check if a benchmark binary exists."""
    path = SCRIPT_DIR / binary
    return path.exists() and path.is_file()


def run_benchmark(
    benchmark: BenchmarkExpectation,
    profiler: Metrix,
    num_replays: int = 1,
    verbose: bool = False,
    raw_counters: bool = False,
) -> Dict[str, Any]:
    """Run a single benchmark and evaluate expectations.

    Returns a result dict with:
        - name: benchmark name
        - status: "pass" | "fail" | "skip" | "error"
        - results: list of expectation evaluation results
        - raw_metrics: dict of all metric values
        - duration_s: wall-clock time
        - error: error message if status is "error"
    """
    result = {
        "name": benchmark.name,
        "description": benchmark.description,
        "binary": benchmark.binary,
        "args": benchmark.args,
        "status": "unknown",
        "results": [],
        "raw_metrics": {},
        "duration_s": 0.0,
    }

    binary_path = SCRIPT_DIR / benchmark.binary
    if not check_binary_exists(benchmark.binary):
        result["status"] = "skip"
        result["error"] = f"Binary not found: {binary_path}"
        return result

    # Build command
    cmd = str(binary_path)
    if benchmark.args:
        cmd += " " + " ".join(benchmark.args)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {benchmark.name}")
        print(f"Command: {cmd}")
        print(f"Metrics: {benchmark.metrics}")
        print(f"{'='*60}")

    t0 = time.time()
    try:
        results = profiler.profile(
            command=cmd,
            metrics=benchmark.metrics,
            kernel_filter=benchmark.kernel_filter,
            num_replays=num_replays,
            cwd=str(SCRIPT_DIR),
            timeout_seconds=120,
        )
        result["duration_s"] = time.time() - t0
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["duration_s"] = time.time() - t0
        return result

    # Find kernel results
    if not results.kernels:
        result["status"] = "error"
        result["error"] = "No kernels found in profiling results"
        return result

    # Use first matching kernel (kernel_filter should have narrowed it down)
    kernel = results.kernels[0]

    if verbose:
        print(f"  Kernel: {kernel.name}")
        print(f"  Duration: {kernel.avg_time_us:.2f} us")
        print(f"  Metrics: {len(kernel.metrics)} collected")

    # Extract metric values
    for metric_name, stats in kernel.metrics.items():
        result["raw_metrics"][metric_name] = {
            "avg": stats.avg,
            "min": stats.min,
            "max": stats.max,
        }
        if verbose:
            print(f"    {metric_name}: avg={stats.avg:.4f}, min={stats.min:.4f}, max={stats.max:.4f}")

    # Evaluate expectations
    all_pass = True
    for exp in benchmark.expectations:
        if exp.metric not in kernel.metrics:
            eval_result = {
                "metric": exp.metric,
                "pass": False,
                "message": f"Metric not found in results",
                "actual": None,
                "expected": exp.description,
                "description": exp.description,
            }
        else:
            actual = kernel.metrics[exp.metric].avg
            eval_result = exp.evaluate(actual)

        result["results"].append(eval_result)
        if not eval_result["pass"]:
            all_pass = False

        if verbose:
            status = "PASS" if eval_result["pass"] else "FAIL"
            print(f"  [{status}] {exp.metric}: {eval_result['message']}")

    result["status"] = "pass" if all_pass else "fail"
    return result


def generate_report(
    results: List[Dict],
    arch: str,
    num_replays: int,
) -> str:
    """Generate a markdown report from benchmark results."""
    lines = []
    lines.append("# Metrix Derived Counter Validation Report\n")
    lines.append(f"**Architecture**: {arch}")
    lines.append(f"**Replays per benchmark**: {num_replays}")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skip")

    lines.append("## Summary\n")
    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| PASS | {passed} |")
    lines.append(f"| FAIL | {failed} |")
    lines.append(f"| ERROR | {errors} |")
    lines.append(f"| SKIP | {skipped} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    # Per-benchmark results
    lines.append("## Per-Benchmark Results\n")
    lines.append("| # | Benchmark | Status | Duration (s) | Checks Passed |")
    lines.append("|---|-----------|--------|-------------|---------------|")
    for i, r in enumerate(results, 1):
        status_emoji = {"pass": "PASS", "fail": "FAIL", "error": "ERROR", "skip": "SKIP"}.get(
            r["status"], r["status"]
        )
        checks = r.get("results", [])
        if checks:
            check_pass = sum(1 for c in checks if c.get("pass", False))
            check_str = f"{check_pass}/{len(checks)}"
        else:
            check_str = "-"
        lines.append(
            f"| {i} | {r['name']} | {status_emoji} | {r['duration_s']:.1f} | {check_str} |"
        )
    lines.append("")

    # Detailed results
    lines.append("## Detailed Results\n")
    for r in results:
        lines.append(f"### {r['name']}\n")
        lines.append(f"**Description**: {r.get('description', '')}")
        lines.append(f"**Command**: `{r['binary']} {' '.join(r.get('args', []))}`")
        lines.append(f"**Status**: {r['status'].upper()}")
        if r.get("error"):
            lines.append(f"**Error**: {r['error']}")
        lines.append("")

        if r.get("results"):
            lines.append("| Metric | Expected | Actual | Result |")
            lines.append("|--------|----------|--------|--------|")
            for check in r["results"]:
                actual = f"{check['actual']:.4f}" if check.get("actual") is not None else "N/A"
                expected = check.get("expected", "")
                status = "PASS" if check.get("pass") else "FAIL"
                lines.append(f"| {check['metric']} | {expected} | {actual} | {status} |")
            lines.append("")

        if r.get("raw_metrics"):
            lines.append("<details><summary>Raw metric values</summary>\n")
            lines.append("| Metric | Avg | Min | Max |")
            lines.append("|--------|-----|-----|-----|")
            for metric, vals in sorted(r["raw_metrics"].items()):
                lines.append(
                    f"| {metric} | {vals['avg']:.4f} | {vals['min']:.4f} | {vals['max']:.4f} |"
                )
            lines.append("</details>\n")

    # Known formula divergences section
    lines.append("## Known Formula Divergences\n")
    lines.append("### 1. L1 Hit Rate")
    lines.append("- **IntelliKit**: `(TCP_TOTAL_CACHE_ACCESSES_sum - TCP_TCC_READ_REQ_sum) / TCP_TOTAL_CACHE_ACCESSES_sum`")
    lines.append("- **Omniperf**: Also subtracts write/atomic miss requests from total accesses")
    lines.append("- **Impact**: IntelliKit may overcount L1 hits for write-heavy kernels (benchmark 3: copy)")
    lines.append("")
    lines.append("### 2. LDS Bank Conflicts")
    lines.append("- **IntelliKit**: `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS` (conflicts per instruction)")
    lines.append("- **Omniperf**: `conflicts / (lds_active - conflicts)` (different denominator)")
    lines.append("- **Impact**: Different absolute values; IntelliKit's is simpler to interpret")
    lines.append("")
    lines.append("### 3. Coalescing Efficiency")
    lines.append("- **IntelliKit**: `(SQ_INSTS_VMEM_RD + SQ_INSTS_VMEM_WR) * 16 / TCP_TOTAL_ACCESSES_sum`")
    lines.append("- **Omniperf**: `(TA_TOTAL_WAVEFRONTS * 64) / TCP_TOTAL_ACCESSES_sum`")
    lines.append("- **Impact**: Different numerator; check stride-sweep results")
    lines.append("")
    lines.append("### 4. FLOPS Wavefront Size")
    lines.append("- **IntelliKit**: Hardcoded `64` in total_flops formula")
    lines.append("- **Risk**: Will break on RDNA4 (wavefront_size=32)")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Metrix derived counter validation")
    parser.add_argument("--arch", type=str, default=None, help="GPU architecture (auto-detect if omitted)")
    parser.add_argument("--bench", type=str, default=None, help="Run only this benchmark (by name)")
    parser.add_argument("--num-replays", type=int, default=3, help="Number of profiling replays")
    parser.add_argument("--output", type=str, default=None, help="Output report file (markdown)")
    parser.add_argument("--json-output", type=str, default=None, help="Output raw results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--raw-counters", action="store_true", help="Also collect raw counters")
    args = parser.parse_args()

    # Detect architecture
    arch = args.arch or detect_arch()
    print(f"Architecture: {arch}")

    # Initialize metrix
    profiler = Metrix(arch=arch)
    print(f"Backend: {profiler.backend.__class__.__name__}")
    print(f"Available metrics: {len(profiler.list_metrics())}")

    # Get benchmark expectations
    benchmarks = get_expectations(arch=arch)
    print(f"Total benchmarks: {len(benchmarks)}")

    # Filter if requested
    if args.bench:
        benchmarks = [b for b in benchmarks if args.bench in b.name]
        if not benchmarks:
            print(f"No benchmarks matching '{args.bench}'")
            sys.exit(1)
        print(f"Running {len(benchmarks)} benchmark(s) matching '{args.bench}'")

    # Check binaries exist
    missing = [b.name for b in benchmarks if not check_binary_exists(b.binary)]
    if missing:
        print(f"\nWARNING: Missing binaries for: {', '.join(missing)}")
        print("Run 'make' in the benchmarks directory first.\n")

    # Run benchmarks
    all_results = []
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] {benchmark.name}...", end=" ", flush=True)
        result = run_benchmark(
            benchmark, profiler, num_replays=args.num_replays,
            verbose=args.verbose, raw_counters=args.raw_counters,
        )
        all_results.append(result)

        status = result["status"].upper()
        if result.get("results"):
            check_pass = sum(1 for c in result["results"] if c.get("pass", False))
            print(f"{status} ({check_pass}/{len(result['results'])} checks)")
        else:
            print(f"{status}")

    # Summary
    print(f"\n{'='*60}")
    passed = sum(1 for r in all_results if r["status"] == "pass")
    failed = sum(1 for r in all_results if r["status"] == "fail")
    errors = sum(1 for r in all_results if r["status"] == "error")
    skipped = sum(1 for r in all_results if r["status"] == "skip")
    total = len(all_results)
    print(f"Results: {passed} pass, {failed} fail, {errors} error, {skipped} skip / {total} total")

    # Print failures
    if failed > 0 or errors > 0:
        print(f"\nFailed/Error benchmarks:")
        for r in all_results:
            if r["status"] in ("fail", "error"):
                print(f"  {r['name']}: {r['status']}")
                if r.get("error"):
                    print(f"    Error: {r['error']}")
                for check in r.get("results", []):
                    if not check.get("pass", False):
                        print(f"    FAIL: {check['metric']}: {check.get('message', '')}")

    # Generate report
    report = generate_report(all_results, arch, args.num_replays)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport written to {args.output}")
    else:
        # Write to default location
        report_path = SCRIPT_DIR / "validation_report.md"
        report_path.write_text(report)
        print(f"\nReport written to {report_path}")

    # JSON output
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(all_results, indent=2, default=str))
        print(f"JSON results written to {args.json_output}")

    # Exit code
    sys.exit(0 if failed == 0 and errors == 0 else 1)


if __name__ == "__main__":
    main()
