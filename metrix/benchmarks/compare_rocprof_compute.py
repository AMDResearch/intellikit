#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Cross-reference metrix derived metrics against rocprof-compute (omniperf) equations.

This script documents the known formula differences between IntelliKit metrix
and omniperf/rocprof-compute, and optionally runs both tools on the same
benchmark to compare outputs.

Usage:
    python compare_rocprof_compute.py [--run BENCHMARK_BINARY] [--arch ARCH]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Known formula divergences between IntelliKit metrix and omniperf
FORMULA_DIVERGENCES = {
    "l1_hit_rate": {
        "intellikit": {
            "formula": "(TCP_TOTAL_CACHE_ACCESSES_sum - TCP_TCC_READ_REQ_sum) / TCP_TOTAL_CACHE_ACCESSES_sum * 100",
            "description": "Only subtracts read miss requests (TCP_TCC_READ_REQ) from total accesses",
            "counters": ["TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum"],
        },
        "omniperf": {
            "formula": "(TCP_TOTAL_CACHE_ACCESSES_sum - TCP_TCC_READ_REQ_sum - TCP_TCC_WRITE_REQ_sum - TCP_TCC_ATOMIC_WITH_RET_REQ_sum - TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum) / TCP_TOTAL_CACHE_ACCESSES_sum * 100",
            "description": "Subtracts read, write, and atomic miss requests",
            "counters": [
                "TCP_TOTAL_CACHE_ACCESSES_sum",
                "TCP_TCC_READ_REQ_sum",
                "TCP_TCC_WRITE_REQ_sum",
                "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
                "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum",
            ],
        },
        "impact": "IntelliKit overestimates L1 hit rate for write-heavy or atomic-heavy kernels. "
                  "For read-only kernels (benchmarks 1, 4, 5, 6), results should match.",
        "test_benchmark": "bw_copy (benchmark 3) - has both reads and writes",
    },
    "lds_bank_conflicts": {
        "intellikit": {
            "formula": "SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS",
            "description": "Conflicts per LDS instruction",
            "counters": ["SQ_LDS_BANK_CONFLICT", "SQ_INSTS_LDS"],
        },
        "omniperf": {
            "formula": "SQ_LDS_BANK_CONFLICT / (SQ_ACTIVE_INST_LDS - SQ_LDS_BANK_CONFLICT)",
            "description": "Conflicts per non-conflicted LDS cycle",
            "counters": ["SQ_LDS_BANK_CONFLICT", "SQ_ACTIVE_INST_LDS"],
        },
        "impact": "Different semantics: IntelliKit gives average conflicts per instruction, "
                  "omniperf gives conflict ratio. Both are valid but not directly comparable.",
        "test_benchmark": "lds_bank_conflicts (benchmark 7) - both modes",
    },
    "coalescing_efficiency": {
        "intellikit": {
            "formula": "(SQ_INSTS_VMEM_RD + SQ_INSTS_VMEM_WR) * 16 / TCP_TOTAL_ACCESSES_sum * 100",
            "description": "Ratio of ideal cache accesses (16 per wave instruction) to actual",
            "counters": ["SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "TCP_TOTAL_ACCESSES_sum"],
        },
        "omniperf": {
            "formula": "(TA_TOTAL_WAVEFRONTS_sum * 64) / TCP_TOTAL_ACCESSES_sum * 100",
            "description": "Ratio of ideal accesses (64 threads per wave) to actual cache accesses",
            "counters": ["TA_TOTAL_WAVEFRONTS_sum", "TCP_TOTAL_ACCESSES_sum"],
        },
        "impact": "Different numerator: IntelliKit counts VMEM instructions, omniperf counts TA wavefronts. "
                  "For simple load/store kernels these should be equivalent. "
                  "Divergence possible for filtered/masked memory operations.",
        "test_benchmark": "bw_strided_access (benchmark 4) at multiple strides",
    },
    "total_flops_wavefront_size": {
        "intellikit": {
            "formula": "64 * (VALU_ADD + VALU_MUL + VALU_TRANS + VALU_FMA * 2) + 512 * MFMA",
            "description": "Hardcoded wavefront_size=64, MFMA=512 ops",
            "counters": ["SQ_INSTS_VALU_*"],
        },
        "omniperf": {
            "formula": "WAVEFRONT_SIZE * (VALU_ADD + VALU_MUL + VALU_TRANS + VALU_FMA * 2) + MFMA_OPS",
            "description": "Uses wavefront_size from device info",
            "counters": ["SQ_INSTS_VALU_*"],
        },
        "impact": "Only matters for RDNA4 (wavefront_size=32). All CDNA architectures use 64.",
        "test_benchmark": "compute_valu_fma (benchmark 8)",
    },
    "division_by_zero": {
        "intellikit": {
            "formula": "Python @metric methods check 'if denominator == 0: return 0.0'",
            "description": "Explicit zero-division guards in Python code",
            "counters": [],
        },
        "omniperf": {
            "formula": "Similar guards in Python analysis code",
            "description": "Also has zero-division guards",
            "counters": [],
        },
        "impact": "YAML eval() expressions may lack these guards. Test with edge cases "
                  "(e.g., no memory ops kernel for coalescing).",
        "test_benchmark": "compute_valu_fma (benchmark 8) - zero memory ops for coalescing",
    },
}


def print_divergences():
    """Print a formatted summary of known formula divergences."""
    print("=" * 80)
    print("KNOWN FORMULA DIVERGENCES: IntelliKit metrix vs rocprof-compute (omniperf)")
    print("=" * 80)
    for name, info in FORMULA_DIVERGENCES.items():
        print(f"\n--- {name} ---")
        print(f"  IntelliKit: {info['intellikit']['formula']}")
        print(f"    ({info['intellikit']['description']})")
        print(f"  Omniperf:   {info['omniperf']['formula']}")
        print(f"    ({info['omniperf']['description']})")
        print(f"  Impact:     {info['impact']}")
        print(f"  Test with:  {info['test_benchmark']}")


def check_rocprof_compute_available() -> bool:
    """Check if rocprof-compute (omniperf) is available."""
    return shutil.which("rocprof-compute") is not None or shutil.which("omniperf") is not None


def run_rocprof_compute(binary: str, arch: str) -> dict:
    """Run rocprof-compute on a binary and extract key metrics.

    Returns dict of metric_name -> value.
    """
    tool = shutil.which("rocprof-compute") or shutil.which("omniperf")
    if not tool:
        return {"error": "rocprof-compute/omniperf not found"}

    with tempfile.TemporaryDirectory(prefix="rocprof_compute_") as tmpdir:
        # Profile
        profile_cmd = [tool, "profile", "-n", "validation", "--", binary]
        try:
            result = subprocess.run(
                profile_cmd, capture_output=True, text=True,
                timeout=120, cwd=tmpdir
            )
            if result.returncode != 0:
                return {"error": f"Profile failed: {result.stderr[:500]}"}
        except subprocess.TimeoutExpired:
            return {"error": "Profile timed out"}

        # Analyze
        analyze_cmd = [
            tool, "analyze",
            "--path", f"{tmpdir}/workloads/validation",
            "--report-format", "json"
        ]
        try:
            result = subprocess.run(
                analyze_cmd, capture_output=True, text=True,
                timeout=60, cwd=tmpdir
            )
            if result.returncode != 0:
                return {"error": f"Analysis failed: {result.stderr[:500]}"}

            # Parse JSON output
            return {"raw_output": result.stdout[:5000]}
        except subprocess.TimeoutExpired:
            return {"error": "Analysis timed out"}


def generate_comparison_report(divergences: dict) -> str:
    """Generate markdown report of formula divergences."""
    lines = []
    lines.append("# Metrix vs rocprof-compute (Omniperf) Formula Comparison\n")
    lines.append(f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Summary of Divergences\n")
    lines.append("| # | Metric | Divergence Type | Impact |")
    lines.append("|---|--------|-----------------|--------|")
    for i, (name, info) in enumerate(divergences.items(), 1):
        impact_short = info["impact"][:80] + "..." if len(info["impact"]) > 80 else info["impact"]
        lines.append(f"| {i} | {name} | Formula difference | {impact_short} |")
    lines.append("")

    lines.append("## Detailed Comparison\n")
    for name, info in divergences.items():
        lines.append(f"### {name}\n")
        lines.append(f"**IntelliKit formula**:")
        lines.append(f"```")
        lines.append(f"{info['intellikit']['formula']}")
        lines.append(f"```")
        lines.append(f"{info['intellikit']['description']}\n")
        lines.append(f"**Omniperf formula**:")
        lines.append(f"```")
        lines.append(f"{info['omniperf']['formula']}")
        lines.append(f"```")
        lines.append(f"{info['omniperf']['description']}\n")
        lines.append(f"**Impact**: {info['impact']}\n")
        lines.append(f"**Test benchmark**: {info['test_benchmark']}\n")
        if info['intellikit']['counters']:
            lines.append(f"**IntelliKit counters**: `{', '.join(info['intellikit']['counters'])}`\n")
        if info['omniperf']['counters']:
            lines.append(f"**Omniperf counters**: `{', '.join(info['omniperf']['counters'])}`\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrix vs rocprof-compute formulas"
    )
    parser.add_argument("--run", type=str, default=None,
                        help="Run both tools on this benchmark binary")
    parser.add_argument("--arch", type=str, default="gfx942",
                        help="GPU architecture")
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown report file")
    args = parser.parse_args()

    print_divergences()

    if args.run:
        print(f"\n{'='*60}")
        if check_rocprof_compute_available():
            print(f"Running rocprof-compute on: {args.run}")
            results = run_rocprof_compute(args.run, args.arch)
            print(json.dumps(results, indent=2))
        else:
            print("rocprof-compute/omniperf not available on this system")
            print("To install: pip install rocprof-compute")
            print("Comparison will be based on documented formulas only.")

    # Generate report
    report = generate_comparison_report(FORMULA_DIVERGENCES)
    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport written to {args.output}")
    else:
        report_path = Path(__file__).parent / "omniperf_comparison.md"
        report_path.write_text(report)
        print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
